//! Model Checker - BFS state exploration with invariant checking
//!
//! This module implements the core model checking algorithm:
//! 1. Generate initial states using Init predicate
//! 2. BFS explore states using Next relation
//! 3. Check invariants at each state
//! 4. Generate counterexample traces when violations found
//!
//! # Algorithm
//!
//! ```text
//! seen = {}
//! queue = [all states satisfying Init]
//! for each s in queue:
//!     if s in seen: continue
//!     seen.add(s)
//!     for each invariant I:
//!         if not I(s): return Error(trace to s)
//!     for each successor t of s via Next:
//!         if t not in seen:
//!             queue.append(t)
//! return Success(|seen| states explored)
//! ```

use crate::arena::{BulkStateHandle, BulkStateStorage};
use crate::compiled_guard::{compile_guard, compile_guard_for_filter, CompiledGuard};
use crate::config::{Config, TerminalSpec};
use crate::constants::bind_constants_from_config;
use crate::coverage::{detect_actions, CoverageStats, DetectedAction};
use crate::enumerate::LocalScope;
use crate::enumerate::{
    enumerate_constraints_to_bulk, enumerate_states_from_constraint_branches,
    enumerate_successors, extract_conjunction_remainder, extract_init_constraints,
    find_unconstrained_vars, print_enum_profile_stats,
};
use crate::error::EvalError;
use crate::eval::{Env, EvalCtx};
use crate::fingerprint::FP64_INIT;
use crate::liveness::{AstToLive, LiveExpr, LivenessChecker, LivenessResult};
use crate::spec_formula::{extract_all_fairness, extract_spec_formula, FairnessConstraint};
use crate::state::{
    print_symmetry_stats, value_fingerprint, ArrayState, DiffSuccessor, Fingerprint, State,
};
use crate::storage::{
    CapacityStatus, FingerprintSet, FingerprintStorage, TraceLocationStorage, TraceLocationsStorage,
};
use crate::trace_file::TraceFile;
use crate::Value;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tla_core::ast::{Expr, Module, OperatorDef, Unit};
use tla_core::span::Span;
use tla_core::{Spanned, SyntaxElement, SyntaxKind, SyntaxNode};

/// Type alias for successor witness map used in symmetry-aware liveness checking.
/// Maps each state fingerprint to its concrete successor states (fingerprint, state pairs).
/// Required because symmetry reduction canonicalizes fingerprints but liveness checking
/// needs actual state values to evaluate ENABLED and action predicates.
pub type SuccessorWitnessMap = FxHashMap<Fingerprint, Vec<(Fingerprint, State)>>;

// Cached debug flags to avoid env::var syscalls in hot paths
fn debug_states() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_DEBUG_STATES").is_ok())
}

fn debug_invariants() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_DEBUG_INVARIANTS").is_ok())
}

fn debug_successors() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_DEBUG_SUCCESSORS").is_ok())
}

fn debug_successors_tlc_fp() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_DEBUG_SUCCESSORS_TLCFP").is_ok())
}

fn debug_successors_limit() -> Option<usize> {
    static LIMIT: OnceLock<Option<usize>> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("TLA2_DEBUG_SUCCESSORS_LIMIT")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
    })
}

fn debug_successors_filter_state_fp() -> Option<Fingerprint> {
    static FILTER: OnceLock<Option<Fingerprint>> = OnceLock::new();
    *FILTER.get_or_init(|| parse_u64_env("TLA2_DEBUG_SUCCESSORS_STATE").map(Fingerprint))
}

fn debug_successors_filter_state_tlc_fp() -> Option<u64> {
    static FILTER: OnceLock<Option<u64>> = OnceLock::new();
    *FILTER.get_or_init(|| parse_u64_env("TLA2_DEBUG_SUCCESSORS_TLC_STATE"))
}

fn debug_successors_dump_state() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_DEBUG_SUCCESSORS_DUMP_STATE").is_ok())
}

fn parse_u64_env(var: &str) -> Option<u64> {
    let raw = std::env::var(var).ok()?;
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    let s_no_prefix = s.strip_prefix("0x").unwrap_or(s);
    // Prefer hex when explicitly prefixed or when it contains hex digits a-f/A-F.
    let looks_hex = s.starts_with("0x")
        || s_no_prefix
            .chars()
            .any(|c| matches!(c, 'a'..='f' | 'A'..='F'));
    if looks_hex {
        u64::from_str_radix(s_no_prefix, 16).ok()
    } else {
        s_no_prefix.parse::<u64>().ok()
    }
}

fn tlc_fp_for_state_values(values: &[Value]) -> u64 {
    let mut fp = FP64_INIT;
    for v in values {
        fp = v.fingerprint_extend(fp);
    }
    fp
}

static DEBUG_SUCCESSOR_LINES: AtomicUsize = AtomicUsize::new(0);

fn should_print_successor_debug_line(force: bool) -> bool {
    if force {
        return true;
    }
    let Some(limit) = debug_successors_limit() else {
        return true;
    };
    let line = DEBUG_SUCCESSOR_LINES.fetch_add(1, AtomicOrdering::Relaxed);
    line < limit
}

fn should_debug_successors_for_state(fp: Fingerprint, tlc_fp: Option<u64>) -> bool {
    debug_successors_filter_state_fp().is_some_and(|target| target == fp)
        || debug_successors_filter_state_tlc_fp().is_some_and(|target| tlc_fp == Some(target))
}

fn skip_liveness() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_SKIP_LIVENESS").is_ok())
}

fn profile_enum() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("TLA2_PROFILE_ENUM").is_ok())
}

/// Check if an expression contains the ENABLED operator.
///
/// ENABLED cannot be evaluated via the normal eval() path - it requires
/// knowledge of successor states and must go through the liveness checker.
/// This function is used to route ENABLED-containing invariants to the
/// liveness checker instead of the standard invariant check path.
fn contains_enabled(expr: &Expr) -> bool {
    match expr {
        Expr::Enabled(_) => true,
        Expr::And(l, r) | Expr::Or(l, r) | Expr::Implies(l, r) | Expr::Equiv(l, r) => {
            contains_enabled(&l.node) || contains_enabled(&r.node)
        }
        Expr::Not(e) => contains_enabled(&e.node),
        Expr::Apply(op, args) => {
            contains_enabled(&op.node) || args.iter().any(|a| contains_enabled(&a.node))
        }
        Expr::Forall(_, body) | Expr::Exists(_, body) => contains_enabled(&body.node),
        Expr::If(cond, then_e, else_e) => {
            contains_enabled(&cond.node)
                || contains_enabled(&then_e.node)
                || contains_enabled(&else_e.node)
        }
        Expr::Let(defs, body) => {
            defs.iter().any(|d| contains_enabled(&d.body.node)) || contains_enabled(&body.node)
        }
        Expr::Always(inner) | Expr::Eventually(inner) => contains_enabled(&inner.node),
        Expr::Case(arms, other) => {
            arms.iter()
                .any(|arm| contains_enabled(&arm.guard.node) || contains_enabled(&arm.body.node))
                || other.as_ref().is_some_and(|e| contains_enabled(&e.node))
        }
        _ => false,
    }
}

/// Result of model checking
#[derive(Debug, Clone)]
pub enum CheckResult {
    /// All reachable states satisfy all invariants
    Success(CheckStats),
    /// An invariant was violated
    InvariantViolation {
        invariant: String,
        trace: Trace,
        stats: CheckStats,
    },
    /// A temporal property was violated by a finite trace
    ///
    /// This is used for safety-style temporal properties (e.g., Init /\ []Action)
    /// that can be checked without the full liveness tableau algorithm.
    PropertyViolation {
        property: String,
        trace: Trace,
        stats: CheckStats,
    },
    /// A liveness property was violated
    LivenessViolation {
        /// The property that was violated
        property: String,
        /// Prefix of the counterexample (finite path to the cycle)
        prefix: Trace,
        /// The cycle itself (lasso)
        cycle: Trace,
        /// Statistics
        stats: CheckStats,
    },
    /// Deadlock detected (state with no successors, if deadlock checking enabled)
    Deadlock { trace: Trace, stats: CheckStats },
    /// Error during checking (evaluation error, missing definition, etc.)
    Error {
        error: CheckError,
        stats: CheckStats,
    },
    /// Exploration limit reached (state or depth limit)
    LimitReached {
        /// Which limit was hit
        limit_type: LimitType,
        /// Statistics at the time limit was reached
        stats: CheckStats,
    },
}

/// Type of exploration limit that was reached
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitType {
    /// Maximum states limit was reached
    States,
    /// Maximum depth limit was reached
    Depth,
}

/// Result of successor generation with information about constraint filtering.
///
/// TLC semantics: A state at the "edge" of the constrained state space (where all
/// successors violate state constraints) is NOT considered a deadlock. Only states
/// with truly no successors (before constraint filtering) are deadlocks.
struct SuccessorResult<T> {
    /// The successors after filtering by state/action constraints
    successors: T,
    /// True if there were any successors before constraint filtering.
    /// Used for deadlock detection: deadlock only if this is false AND successors is empty.
    had_raw_successors: bool,
}

enum SafetyTemporalPropertyOutcome {
    NotApplicable,
    Satisfied,
    Violated(Box<CheckResult>),
}

struct SafetyTemporalProperty {
    init_terms: Vec<Spanned<Expr>>,
    always_terms: Vec<Spanned<Expr>>,
}

/// Safety parts extracted from a mixed safety/liveness property.
///
/// Used by `separate_safety_liveness_parts` to split properties like
/// `Init /\ [][Next]_vars /\ Liveness` into checkable parts.
struct PropertySafetyParts {
    /// Init-level terms (checked on initial states)
    init_terms: Vec<Spanned<Expr>>,
    /// Always-action terms (checked on transitions): body of `[]Action` where Action has no temporal
    always_terms: Vec<Spanned<Expr>>,
}

/// Statistics from model checking
#[derive(Debug, Clone, Default)]
pub struct CheckStats {
    /// Number of distinct states explored
    pub states_found: usize,
    /// Number of initial states
    pub initial_states: usize,
    /// Maximum queue depth (BFS frontier)
    pub max_queue_depth: usize,
    /// Number of transitions examined
    pub transitions: usize,
    /// Maximum BFS depth reached
    pub max_depth: usize,
    /// Detected action names from Next relation (top-level disjuncts)
    pub detected_actions: Vec<String>,
    /// Optional coverage statistics (enabled via `ModelChecker::set_collect_coverage(true)`)
    pub coverage: Option<CoverageStats>,
}

/// Progress information reported during model checking
#[derive(Debug, Clone)]
pub struct Progress {
    /// Number of distinct states found so far
    pub states_found: usize,
    /// Current BFS depth being explored
    pub current_depth: usize,
    /// Current queue size (BFS frontier)
    pub queue_size: usize,
    /// Number of transitions explored so far
    pub transitions: usize,
    /// Seconds elapsed since model checking started
    pub elapsed_secs: f64,
    /// States explored per second (0.0 if elapsed is 0)
    pub states_per_sec: f64,
}

/// Type alias for progress callback function
pub type ProgressCallback = Box<dyn Fn(&Progress) + Send + Sync>;

/// An error during model checking
#[derive(Debug, Clone)]
pub enum CheckError {
    /// Missing Init definition
    MissingInit,
    /// Missing Next definition
    MissingNext,
    /// Missing invariant definition
    MissingInvariant(String),
    /// Missing property definition
    MissingProperty(String),
    /// Evaluation error
    EvalError(EvalError),
    /// Init predicate didn't return boolean
    InitNotBoolean,
    /// Next relation didn't return boolean
    NextNotBoolean,
    /// Invariant didn't return boolean
    InvariantNotBoolean(String),
    /// Property didn't return boolean
    PropertyNotBoolean(String),
    /// No variables in spec
    NoVariables,
    /// Init predicate contains expressions that cannot be enumerated
    InitCannotEnumerate(String),
    /// SPECIFICATION directive error
    SpecificationError(String),
    /// Liveness checking error
    LivenessError(String),
    /// Fingerprint storage overflow - some states were dropped
    FingerprintStorageOverflow {
        /// Number of fingerprints that were dropped
        dropped: usize,
    },
    /// ASSUME statement evaluated to FALSE
    AssumeFalse {
        /// Location information for the assumption (formatted as "line X, col Y to line X2, col Y2 of module M")
        location: String,
    },
}

impl std::fmt::Display for CheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckError::MissingInit => write!(f, "Missing INIT definition"),
            CheckError::MissingNext => write!(f, "Missing NEXT definition"),
            CheckError::MissingInvariant(name) => write!(f, "Missing invariant: {}", name),
            CheckError::MissingProperty(name) => write!(f, "Missing property: {}", name),
            CheckError::EvalError(e) => write!(f, "Evaluation error: {}", e),
            CheckError::InitNotBoolean => write!(f, "Init predicate must return boolean"),
            CheckError::NextNotBoolean => write!(f, "Next relation must return boolean"),
            CheckError::InvariantNotBoolean(name) => {
                write!(f, "Invariant {} must return boolean", name)
            }
            CheckError::PropertyNotBoolean(name) => {
                write!(f, "Property {} must return boolean", name)
            }
            CheckError::NoVariables => write!(f, "Specification has no variables"),
            CheckError::InitCannotEnumerate(hint) => {
                write!(f, "Cannot enumerate states from Init predicate: {}", hint)
            }
            CheckError::SpecificationError(msg) => {
                write!(f, "SPECIFICATION error: {}", msg)
            }
            CheckError::LivenessError(msg) => {
                write!(f, "Liveness error: {}", msg)
            }
            CheckError::FingerprintStorageOverflow { dropped } => {
                write!(
                    f,
                    "Fingerprint storage overflow: {} states were dropped. \
                     Results may be incomplete. Increase --mmap-fingerprints capacity.",
                    dropped
                )
            }
            CheckError::AssumeFalse { location } => {
                write!(f, "Assumption {} is false.", location)
            }
        }
    }
}

impl std::error::Error for CheckError {}

impl From<EvalError> for CheckError {
    fn from(e: EvalError) -> Self {
        CheckError::EvalError(e)
    }
}

/// Format a Span's location for error reporting
///
/// Returns a string like "bytes 123-456 of module M" for displaying
/// where an ASSUME statement is located. TLC uses "line X, col Y to line X2, col Y2"
/// but we don't have source text available here to compute line/column.
fn format_span_location(span: &Span, module_name: &str) -> String {
    format!(
        "bytes {}-{} of module {}",
        span.start, span.end, module_name
    )
}

/// A counterexample trace - sequence of states leading to the error
#[derive(Debug, Clone)]
pub struct Trace {
    /// Sequence of states from initial to error state
    pub states: Vec<State>,
}

impl Trace {
    /// Create an empty trace
    pub fn new() -> Self {
        Trace { states: Vec::new() }
    }

    /// Create a trace from states
    pub fn from_states(states: Vec<State>) -> Self {
        Trace { states }
    }

    /// Length of the trace
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if trace is empty
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, state) in self.states.iter().enumerate() {
            writeln!(f, "State {} ({}):", i + 1, state.fingerprint())?;
            for (name, value) in state.vars() {
                writeln!(f, "  {} = {}", name, value)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Synthetic operator name used for inline NEXT expressions.
/// When a spec formula contains something like `Init /\ [][\E n \in Node: Next(n)]_vars`,
/// we create a synthetic operator with this name to hold the lowered expression.
pub const INLINE_NEXT_NAME: &str = "__TLA2_INLINE_NEXT__";

/// Resolved Init and Next names from config (either direct or via SPECIFICATION)
#[derive(Debug, Clone)]
pub struct ResolvedSpec {
    /// Init predicate name
    pub init: String,
    /// Next relation name (may be an operator name or an inline expression text)
    pub next: String,
    /// The syntax node for inline NEXT expressions (e.g., `\E n \in Node: Next(n)`)
    /// When present, the checker should create a synthetic operator from this node.
    pub next_node: Option<SyntaxNode>,
    /// Fairness constraints extracted from SPEC formula (if any)
    pub fairness: Vec<FairnessConstraint>,
    /// Whether the spec uses `[A]_v` form (stuttering allowed) or `<<A>>_v` form (stuttering forbidden).
    /// When `true`, the spec explicitly permits stuttering (staying in the same state), so deadlock
    /// (no enabled actions) is NOT a violation - the system can legally stutter forever.
    /// When `false`, the spec uses `<<A>>_v` which requires actual progress, so deadlock IS a violation.
    /// Defaults to `true` (stuttering allowed) which is the most common TLA+ pattern.
    pub stuttering_allowed: bool,
}

/// Resolve Init/Next from config, potentially using SPECIFICATION directive.
///
/// If config has explicit INIT and NEXT, use those directly.
/// If config has SPECIFICATION, find the operator body in the syntax tree
/// and extract Init/Next from the temporal formula.
///
/// # Arguments
/// * `config` - The TLC configuration
/// * `syntax_tree` - The syntax tree from parsing (needed for SPECIFICATION extraction)
///
/// # Returns
/// * `Ok(ResolvedSpec)` with init and next names
/// * `Err(CheckError)` if resolution fails
pub fn resolve_spec_from_config(
    config: &Config,
    syntax_tree: &SyntaxNode,
) -> Result<ResolvedSpec, CheckError> {
    resolve_spec_from_config_with_extends(config, syntax_tree, &[])
}

/// Resolve Init/Next from config, searching both main and extended module syntax trees.
///
/// This is useful when the SPECIFICATION operator is defined in an extended module.
pub fn resolve_spec_from_config_with_extends(
    config: &Config,
    syntax_tree: &SyntaxNode,
    extended_trees: &[&SyntaxNode],
) -> Result<ResolvedSpec, CheckError> {
    // If explicit INIT/NEXT provided, use them
    if let (Some(init), Some(next)) = (&config.init, &config.next) {
        return Ok(ResolvedSpec {
            init: init.clone(),
            next: next.clone(),
            next_node: None,          // Explicit INIT/NEXT are always operator names
            fairness: Vec::new(),     // No fairness when using explicit INIT/NEXT
            stuttering_allowed: true, // Default to stuttering allowed when using explicit INIT/NEXT
        });
    }

    // Try to extract from SPECIFICATION
    if let Some(spec_name) = &config.specification {
        return resolve_from_specification_multi(spec_name, syntax_tree, extended_trees);
    }

    // No way to determine Init/Next
    if config.init.is_none() {
        return Err(CheckError::MissingInit);
    }
    Err(CheckError::MissingNext)
}

/// Find an operator body in the syntax tree by name
fn find_op_body_in_tree(tree: &SyntaxNode, name: &str) -> Option<SyntaxNode> {
    fn search(node: &SyntaxNode, name: &str) -> Option<SyntaxNode> {
        if node.kind() == SyntaxKind::OperatorDef {
            // Check if this operator has the target name
            let mut found_name = false;
            let mut name_token_index = 0;
            for (i, child) in node.children_with_tokens().enumerate() {
                if let SyntaxElement::Token(t) = child {
                    if t.kind() == SyntaxKind::Ident && t.text() == name {
                        found_name = true;
                        name_token_index = i;
                        break;
                    }
                }
            }
            if found_name {
                // Return the body expression(s).
                //
                // Most operator bodies are represented as a single CST node after `==`.
                // However, conjunction-list style bodies can appear as multiple top-level
                // expression nodes (one per `/\` line). In that case, return the OperatorDef
                // node itself so downstream traversals see the full body.
                let mut body_nodes = Vec::new();
                let mut past_def_eq = false;
                for child in node.children_with_tokens() {
                    match child {
                        SyntaxElement::Token(t) if t.kind() == SyntaxKind::DefEqOp => {
                            past_def_eq = true;
                        }
                        SyntaxElement::Node(n) if past_def_eq => {
                            body_nodes.push(n);
                        }
                        _ => {}
                    }
                }
                match body_nodes.as_slice() {
                    [single] => return Some(single.clone()),
                    [] => {}
                    _ => return Some(node.clone()),
                }
                // If no explicit expression node found, the body might be a bare identifier
                // (e.g., `spec_123 == LSpec`). In this case, return the OperatorDef node itself
                // so the caller can handle the identifier body specially.
                // Look for any Ident token after the == operator that isn't the operator name.
                let mut past_def_eq = false;
                for (i, child) in node.children_with_tokens().enumerate() {
                    if let SyntaxElement::Token(t) = child {
                        if t.kind() == SyntaxKind::DefEqOp {
                            past_def_eq = true;
                            continue;
                        }
                        if past_def_eq && t.kind() == SyntaxKind::Ident && i > name_token_index {
                            // Found a bare identifier body - return the OperatorDef
                            // so the caller can extract and follow the reference
                            return Some(node.clone());
                        }
                    }
                }
            }
        }
        for child in node.children() {
            if let Some(found) = search(&child, name) {
                return Some(found);
            }
        }
        None
    }
    search(tree, name)
}

/// Check if a syntax node contains temporal operators (WF, SF, [], <>, ~>).
/// This is used to detect complex fairness formulas like `\A p: WF_vars(Action(p))`
/// that can't be extracted as simple FairnessConstraints.
fn contains_temporal_operators(node: &SyntaxNode) -> bool {
    fn search(node: &SyntaxNode) -> bool {
        // Check for temporal operator tokens
        for child in node.children_with_tokens() {
            match child {
                SyntaxElement::Token(t) => {
                    match t.kind() {
                        // Temporal operator keywords
                        SyntaxKind::WeakFairKw
                        | SyntaxKind::StrongFairKw
                        | SyntaxKind::AlwaysOp
                        | SyntaxKind::EventuallyOp
                        | SyntaxKind::LeadsToOp => return true,
                        // WF_xxx or SF_xxx identifiers (parsed as single tokens)
                        SyntaxKind::Ident => {
                            let text = t.text();
                            if text.starts_with("WF_") || text.starts_with("SF_") {
                                return true;
                            }
                        }
                        _ => {}
                    }
                }
                SyntaxElement::Node(n) => {
                    if search(&n) {
                        return true;
                    }
                }
            }
        }
        false
    }
    search(node)
}

/// Resolve Init/Next from a SPECIFICATION temporal formula, searching multiple trees
fn resolve_from_specification_multi(
    spec_name: &str,
    syntax_tree: &SyntaxNode,
    extended_trees: &[&SyntaxNode],
) -> Result<ResolvedSpec, CheckError> {
    let mut visited = FxHashSet::default();
    resolve_from_specification_multi_rec(spec_name, syntax_tree, extended_trees, &mut visited)
}

fn resolve_from_specification_multi_rec(
    spec_name: &str,
    syntax_tree: &SyntaxNode,
    extended_trees: &[&SyntaxNode],
    visited: &mut FxHashSet<String>,
) -> Result<ResolvedSpec, CheckError> {
    if !visited.insert(spec_name.to_string()) {
        return Err(CheckError::SpecificationError(format!(
            "cyclic SPECIFICATION reference involving '{}'",
            spec_name
        )));
    }

    // Find operator body in main module first, then extended modules.
    let spec_body = find_op_body_in_tree(syntax_tree, spec_name).or_else(|| {
        extended_trees
            .iter()
            .find_map(|ext_tree| find_op_body_in_tree(ext_tree, spec_name))
    });

    let Some(spec_body) = spec_body else {
        return Err(CheckError::SpecificationError(format!(
            "operator '{}' not found",
            spec_name
        )));
    };

    // Extract Init/Next from the temporal formula, if it matches a known pattern.
    if let Some(formula) = extract_spec_formula(&spec_body) {
        // Even when the core Init /\ [][Next]_vars pattern matches, the spec body can
        // still conjunct in fairness/liveness assumptions via referenced operators:
        //
        //   Spec == Init /\ [][Next]_vars /\ Liveness
        //
        // If we return immediately here, we'd miss fairness from `Liveness` and get
        // false-positive liveness violations (e.g., SchedulingAllocator).
        let mut fairness = formula.fairness;

        // Extract fairness from any referenced operators at this level.
        // Only include candidates that are actually defined operators.
        let mut candidates = Vec::new();
        let mut seen = FxHashSet::default();
        for element in spec_body.descendants_with_tokens() {
            if let SyntaxElement::Token(t) = element {
                if t.kind() == SyntaxKind::Ident {
                    let name = t.text().to_string();
                    if seen.insert(name.clone()) {
                        candidates.push(name);
                    }
                }
            }
        }

        for candidate in &candidates {
            // Skip core components of the spec formula.
            if candidate == &formula.init || candidate == &formula.next {
                continue;
            }
            if let Some(vars) = &formula.vars {
                if candidate == vars {
                    continue;
                }
            }

            let candidate_body = find_op_body_in_tree(syntax_tree, candidate).or_else(|| {
                extended_trees
                    .iter()
                    .find_map(|ext_tree| find_op_body_in_tree(ext_tree, candidate))
            });

            let Some(body) = candidate_body else {
                continue;
            };

            let candidate_fairness = extract_all_fairness(&body);
            if candidate_fairness.is_empty() {
                if contains_temporal_operators(&body) {
                    fairness.push(FairnessConstraint::TemporalRef {
                        op_name: candidate.clone(),
                    });
                }
            } else {
                fairness.extend(candidate_fairness);
            }
        }

        // When there's an inline NEXT expression, use a synthetic operator name
        // The actual operator will be created by ModelChecker::register_inline_next()
        let next_name = if formula.next_node.is_some() {
            INLINE_NEXT_NAME.to_string()
        } else {
            formula.next
        };

        return Ok(ResolvedSpec {
            init: formula.init,
            next: next_name,
            next_node: formula.next_node,
            fairness,
            stuttering_allowed: formula.stuttering_allowed,
        });
    }

    // Fallback: attempt to resolve SPECIFICATION operators that wrap the actual temporal
    // formula in conjunctions for side effects (e.g., `TestSpec == PrintT(R) /\\ Spec`).
    // Try any referenced operators in the body and accept the first one that yields Init/Next.
    //
    // IMPORTANT: Extract fairness constraints from THIS level before recursing.
    // For specs like `SpecWeakFair == Spec /\ WF_vars(Next)`, the fairness constraint
    // is at this level but Init/Next comes from the nested `Spec` reference.
    let local_fairness = extract_all_fairness(&spec_body);

    let mut candidates = Vec::new();
    let mut seen = FxHashSet::default();
    for element in spec_body.descendants_with_tokens() {
        if let SyntaxElement::Token(t) = element {
            if t.kind() == SyntaxKind::Ident {
                let name = t.text().to_string();
                if seen.insert(name.clone()) {
                    candidates.push(name);
                }
            }
        }
    }

    // First pass: find Init/Next from candidates
    let mut result: Option<ResolvedSpec> = None;
    for candidate in &candidates {
        if visited.contains(candidate) {
            continue;
        }

        // Only attempt candidates that are actually defined operators in any tree.
        let has_def = find_op_body_in_tree(syntax_tree, candidate).is_some()
            || extended_trees
                .iter()
                .any(|ext_tree| find_op_body_in_tree(ext_tree, candidate).is_some());
        if !has_def {
            continue;
        }

        if let Ok(resolved) =
            resolve_from_specification_multi_rec(candidate, syntax_tree, extended_trees, visited)
        {
            result = Some(resolved);
            break;
        }
    }

    // Second pass: extract fairness from ALL candidates that contain temporal formulas
    // This handles specs like `LISpec == ISpec /\ Liveness2` where:
    // - ISpec contains Init/Next (safety)
    // - Liveness2 contains fairness (temporal, e.g., \A p: WF_vars(Action(p)))
    let mut all_fairness = local_fairness;
    for candidate in &candidates {
        // Skip if we already visited this candidate during Init/Next resolution
        if visited.contains(candidate) {
            continue;
        }

        // Find the operator body in any tree
        let candidate_body = find_op_body_in_tree(syntax_tree, candidate).or_else(|| {
            extended_trees
                .iter()
                .find_map(|ext_tree| find_op_body_in_tree(ext_tree, candidate))
        });

        if let Some(body) = candidate_body {
            // Extract any fairness constraints from this candidate's body
            let candidate_fairness = extract_all_fairness(&body);
            if candidate_fairness.is_empty() {
                // No simple fairness found. Check if this contains temporal operators
                // (WF, SF, [], <>) which might be inside quantifiers like \A p: WF_vars(Action(p)).
                // If so, store a TemporalRef to convert the full operator at liveness check time.
                if contains_temporal_operators(&body) {
                    all_fairness.push(FairnessConstraint::TemporalRef {
                        op_name: candidate.clone(),
                    });
                }
            } else {
                all_fairness.extend(candidate_fairness);
            }
        }
    }

    if let Some(mut resolved) = result {
        // Merge fairness: collected fairness from all candidates comes first
        all_fairness.extend(resolved.fairness);
        resolved.fairness = all_fairness;
        return Ok(resolved);
    }

    Err(CheckError::SpecificationError(format!(
        "unsupported SPECIFICATION formula in operator '{}'",
        spec_name
    )))
}

/// The model checker
pub struct ModelChecker<'a> {
    /// Configuration
    config: &'a Config,
    /// Evaluation context
    ctx: EvalCtx,
    /// Seen states (fingerprint -> array state for trace reconstruction)
    /// Only populated when `store_full_states` is true
    /// Uses ArrayState instead of State to reduce memory overhead
    seen: FxHashMap<Fingerprint, ArrayState>,
    /// Seen fingerprints (for memory-efficient mode when `store_full_states` is false)
    /// Uses FingerprintSet trait which supports both in-memory HashSet and memory-mapped storage.
    seen_fps: Arc<dyn FingerprintSet>,
    /// Parent pointers for trace reconstruction (child fingerprint -> parent fingerprint)
    /// Only populated when `store_full_states` is true
    parents: FxHashMap<Fingerprint, Fingerprint>,
    /// Depth tracking for each state (fingerprint -> depth)
    depths: FxHashMap<Fingerprint, usize>,
    /// Variable names
    vars: Vec<Arc<str>>,
    /// Whether to check for deadlocks
    check_deadlock: bool,
    /// Maximum states to explore (None = unlimited)
    max_states: Option<usize>,
    /// Maximum BFS depth (None = unlimited)
    max_depth: Option<usize>,
    /// Statistics
    stats: CheckStats,
    /// Cached operator definitions
    op_defs: FxHashMap<String, OperatorDef>,
    /// Progress callback (called every N states)
    progress_callback: Option<ProgressCallback>,
    /// How often to report progress (in states)
    progress_interval: usize,
    /// Cached successors from BFS (fingerprint -> list of successor fingerprints)
    /// Used for liveness checking to avoid regenerating transitions
    cached_successors: FxHashMap<Fingerprint, Vec<Fingerprint>>,
    /// Cached successor states for liveness checking under symmetry.
    ///
    /// With symmetry reduction enabled, `cached_successors` only stores the
    /// canonical successor fingerprints, losing which concrete successor state
    /// witnessed each transition. Liveness checking needs access to concrete
    /// successor state values to correctly evaluate `ENABLED` and action-level
    /// predicates under symmetry (avoids false positives).
    ///
    /// Keyed by the canonical fingerprint of the source state; each entry
    /// stores `(canonical_successor_fp, successor_state)` pairs.
    ///
    /// Only populated when liveness checking is enabled and symmetry is active.
    cached_successor_states: FxHashMap<Fingerprint, Vec<(Fingerprint, ArrayState)>>,
    /// Fairness constraints extracted from SPEC formula
    fairness: Vec<FairnessConstraint>,
    /// Whether to collect per-action coverage statistics
    collect_coverage: bool,
    /// Cached detected actions (including expressions) for coverage collection
    coverage_actions: Vec<DetectedAction>,
    /// Symmetry permutations for state reduction (empty if no SYMMETRY in config)
    symmetry_perms: Vec<crate::value::FuncValue>,
    /// Cache: original fingerprint -> canonical fingerprint (for symmetry reduction)
    /// Avoids recomputing canonical fingerprint for the same state
    symmetry_fp_cache: FxHashMap<Fingerprint, Fingerprint>,
    /// Whether to store full states for trace reconstruction
    /// When false, only fingerprints are stored (saves memory but no trace available)
    store_full_states: bool,
    /// Whether to auto-create a temp trace file when store_full_states is false
    /// Default: true. Set to false for --no-trace mode (no trace reconstruction at all)
    auto_create_trace_file: bool,
    /// Disk-based trace file for large state space exploration
    /// When enabled, stores (predecessor_loc, fingerprint) pairs on disk for trace reconstruction
    trace_file: Option<TraceFile>,
    /// Mapping from fingerprint to trace file location
    /// Uses TraceLocationsStorage for scalable (mmap) or in-memory storage
    trace_locs: TraceLocationsStorage,
    /// Cached Init operator name (for trace reconstruction from fingerprints)
    cached_init_name: Option<String>,
    /// Cached Next operator name (for trace reconstruction from fingerprints)
    cached_next_name: Option<String>,
    /// Cached VIEW operator name (for state abstraction in fingerprinting)
    /// When set, fingerprinting uses this operator's value instead of full state
    cached_view_name: Option<String>,
    /// Directory for saving checkpoints during model checking
    checkpoint_dir: Option<PathBuf>,
    /// Interval between checkpoints (in seconds)
    checkpoint_interval: Duration,
    /// Time of last checkpoint
    last_checkpoint_time: Option<Instant>,
    /// Spec path for checkpoint metadata
    checkpoint_spec_path: Option<String>,
    /// Config path for checkpoint metadata
    checkpoint_config_path: Option<String>,
    /// Last reported capacity status (for suppressing duplicate warnings)
    last_capacity_status: CapacityStatus,
    /// Whether to cache successors for liveness checking
    /// Only true when: config has properties AND skip_liveness() is false AND store_full_states is true
    cache_successors_for_liveness: bool,
    /// Pre-compiled invariants for efficient evaluation
    /// Each invariant is compiled once to avoid AST traversal overhead per state
    compiled_invariants: Vec<(String, CompiledGuard)>,
    /// Invariants that contain ENABLED operator
    /// These must be checked via the liveness checker since ENABLED requires
    /// knowledge of successor states and cannot be evaluated by eval()
    enabled_invariants: Vec<String>,
    /// ASSUME statements collected from all modules (main + extended)
    /// Each entry is (module_name, assume_expr) for error reporting
    assumes: Vec<(String, Spanned<Expr>)>,
}

impl<'a> ModelChecker<'a> {
    /// Create a new model checker
    pub fn new(module: &'a Module, config: &'a Config) -> Self {
        Self::new_with_extends(module, &[], config)
    }

    /// Create a new model checker with extended modules
    ///
    /// The `extended_modules` should be modules that `module` extends (via EXTENDS).
    /// Their operator definitions will be loaded first, then the main module's
    /// definitions (which may override them).
    pub fn new_with_extends(
        module: &'a Module,
        extended_modules: &[&Module],
        config: &'a Config,
    ) -> Self {
        let mut ctx = EvalCtx::new();

        let mut module_by_name: std::collections::HashMap<&str, &Module> =
            std::collections::HashMap::new();
        for ext_mod in extended_modules {
            module_by_name.insert(ext_mod.name.node.as_str(), *ext_mod);
        }

        // Determine which loaded modules should contribute to the *unqualified* operator namespace.
        //
        // In TLC:
        // - `EXTENDS M` imports M's operators (and variable declarations) unqualified.
        // - a standalone `INSTANCE M` statement also imports operators unqualified.
        // - a *named instance* (`I == INSTANCE M WITH ...`) does NOT import M unqualified; it
        //   should only be reachable via `I!Op`.
        //
        // The module loader gives us a superset of modules (extends + instances, including nested
        // instances). We compute the unqualified import closure here and load only those modules
        // into the shared operator environment to avoid name collisions.
        let mut unqualified_modules: std::collections::HashSet<&str> =
            std::collections::HashSet::new();
        let mut stack: Vec<&str> = Vec::new();
        // EXTENDS closure from the main module
        stack.extend(module.extends.iter().map(|s| s.node.as_str()));
        // Standalone INSTANCE imports from the main module
        for unit in &module.units {
            if let Unit::Instance(inst) = &unit.node {
                stack.push(inst.module.node.as_str());
            }
        }
        // Transitive closure: follow EXTENDS and standalone INSTANCE in imported modules.
        while let Some(name) = stack.pop() {
            if !unqualified_modules.insert(name) {
                continue;
            }
            let Some(m) = module_by_name.get(name) else {
                continue;
            };
            stack.extend(m.extends.iter().map(|s| s.node.as_str()));
            for unit in &m.units {
                if let Unit::Instance(inst) = &unit.node {
                    stack.push(inst.module.node.as_str());
                }
            }
        }

        // Load unqualified imported modules first (in order).
        for ext_mod in extended_modules {
            if !unqualified_modules.contains(ext_mod.name.node.as_str()) {
                continue;
            }
            ctx.load_module(ext_mod);
        }
        // Load main module last (can override)
        // This also registers named instances (operators with InstanceExpr body)
        ctx.load_module(module);

        // Load operators for all non-stdlib modules into `instance_ops` so module/instance
        // references (`I!Op`) can be evaluated without importing them unqualified.
        for ext_mod in extended_modules {
            ctx.load_instance_module(ext_mod.name.node.clone(), ext_mod);
        }

        // Extract variable names and operator definitions from all modules
        let mut vars: Vec<Arc<str>> = Vec::new();
        let mut op_defs: FxHashMap<String, OperatorDef> = FxHashMap::default();

        // First from extended modules
        for ext_mod in extended_modules {
            // Only modules imported unqualified (EXTENDS / standalone INSTANCE) should contribute
            // state variables and unqualified operator definitions.
            if !unqualified_modules.contains(ext_mod.name.node.as_str()) {
                continue;
            }
            for unit in &ext_mod.units {
                match &unit.node {
                    Unit::Variable(var_names) => {
                        for var in var_names {
                            if !vars.iter().any(|v| v.as_ref() == var.node.as_str()) {
                                vars.push(Arc::from(var.node.as_str()));
                            }
                        }
                    }
                    Unit::Operator(def) => {
                        op_defs.insert(def.name.node.clone(), def.clone());
                    }
                    _ => {}
                }
            }
        }

        // Then from main module (may shadow)
        for unit in &module.units {
            match &unit.node {
                Unit::Variable(var_names) => {
                    for var in var_names {
                        if !vars.iter().any(|v| v.as_ref() == var.node.as_str()) {
                            vars.push(Arc::from(var.node.as_str()));
                        }
                    }
                }
                Unit::Operator(def) => {
                    op_defs.insert(def.name.node.clone(), def.clone());
                }
                _ => {}
            }
        }

        // Sort variables to ensure fingerprints are consistent with OrdMap-based fingerprinting
        // (OrdMap iterates in sorted key order, so VarRegistry must match)
        vars.sort();

        // Register variables in the VarRegistry for O(1) indexed access in BFS loop
        ctx.register_vars(vars.iter().cloned());

        // Collect ASSUME statements from all modules for pre-checking
        // TLC checks all ASSUME statements before model checking begins
        let mut assumes: Vec<(String, Spanned<Expr>)> = Vec::new();
        for ext_mod in extended_modules {
            // Only collect from modules imported unqualified
            if !unqualified_modules.contains(ext_mod.name.node.as_str()) {
                continue;
            }
            for unit in &ext_mod.units {
                if let Unit::Assume(assume) = &unit.node {
                    // If named, also register as an operator definition
                    if let Some(name) = &assume.name {
                        let op_def = OperatorDef {
                            name: name.clone(),
                            params: vec![],
                            body: assume.expr.clone(),
                            local: false,
                        };
                        op_defs.insert(name.node.clone(), op_def);
                    }
                    assumes.push((ext_mod.name.node.clone(), assume.expr.clone()));
                }
            }
        }
        // Then from main module
        for unit in &module.units {
            if let Unit::Assume(assume) = &unit.node {
                // If named, also register as an operator definition
                if let Some(name) = &assume.name {
                    let op_def = OperatorDef {
                        name: name.clone(),
                        params: vec![],
                        body: assume.expr.clone(),
                        local: false,
                    };
                    op_defs.insert(name.node.clone(), op_def);
                }
                assumes.push((module.name.node.clone(), assume.expr.clone()));
            }
        }

        // Symmetry permutations will be computed lazily after constants are bound
        let symmetry_perms = Vec::new();

        ModelChecker {
            config,
            ctx,
            seen: FxHashMap::default(),
            seen_fps: Arc::new(FingerprintStorage::in_memory()) as Arc<dyn FingerprintSet>,
            parents: FxHashMap::default(),
            depths: FxHashMap::default(),
            vars,
            check_deadlock: config.check_deadlock,
            max_states: None,
            max_depth: None,
            stats: CheckStats::default(),
            op_defs,
            progress_callback: None,
            progress_interval: 1000, // Default: report every 1000 states
            cached_successors: FxHashMap::default(),
            cached_successor_states: FxHashMap::default(),
            fairness: Vec::new(),
            collect_coverage: false,
            coverage_actions: Vec::new(),
            symmetry_perms,
            symmetry_fp_cache: FxHashMap::default(),
            store_full_states: false, // Default: fingerprint-only for 42x memory reduction (see #88)
            auto_create_trace_file: true, // Auto-create temp trace file for reconstruction
            trace_file: None, // Auto-created in check() when auto_create_trace_file is true
            trace_locs: TraceLocationsStorage::in_memory(),
            cached_init_name: None,
            cached_next_name: None,
            cached_view_name: None,
            checkpoint_dir: None,
            checkpoint_interval: Duration::from_secs(300), // Default: 5 minutes
            last_checkpoint_time: None,
            checkpoint_spec_path: None,
            checkpoint_config_path: None,
            last_capacity_status: CapacityStatus::Normal,
            // Cache successors only if we'll actually do liveness checking
            // (has properties, skip_liveness not set, and store_full_states is true)
            // Note: Disabled by default since store_full_states defaults to false (fingerprint-only mode).
            // When full state storage is re-enabled, restore the condition:
            // !config.properties.is_empty() && !skip_liveness() && config.store_full_states
            cache_successors_for_liveness: false,
            compiled_invariants: Vec::new(), // Compiled during check() initialization
            enabled_invariants: Vec::new(),  // Populated during check() initialization
            assumes,
        }
    }

    /// Register an inline NEXT expression from a ResolvedSpec.
    ///
    /// When the SPECIFICATION formula contains an inline NEXT expression like
    /// `Init /\ [][\E n \in Node: Next(n)]_vars`, the `resolved.next_node` contains
    /// the CST node for the expression. This method lowers it to an AST and creates
    /// a synthetic operator definition.
    ///
    /// Call this after creating the checker if `resolved.next_node` is Some.
    pub fn register_inline_next(&mut self, resolved: &ResolvedSpec) -> Result<(), CheckError> {
        let Some(ref node) = resolved.next_node else {
            return Ok(()); // No inline NEXT, nothing to do
        };

        // Lower the CST node to an AST expression
        let expr = tla_core::lower_single_expr(tla_core::FileId(0), node).ok_or_else(|| {
            CheckError::SpecificationError(format!(
                "Failed to lower inline NEXT expression: {}",
                node.text()
            ))
        })?;

        // Create a synthetic operator definition
        let op_def = OperatorDef {
            name: Spanned::dummy(INLINE_NEXT_NAME.to_string()),
            params: vec![],
            body: Spanned::dummy(expr),
            local: false,
        };

        // Register the operator in our definitions and evaluation context
        self.op_defs
            .insert(INLINE_NEXT_NAME.to_string(), op_def.clone());
        self.ctx.define_op(INLINE_NEXT_NAME.to_string(), op_def);

        Ok(())
    }

    /// Compute symmetry permutations from the SYMMETRY configuration
    ///
    /// Evaluates the SYMMETRY operator (if configured) to get the set of permutation
    /// functions. These are used during fingerprinting to identify symmetric states.
    ///
    /// Computes the group closure by composing all pairs of permutations until
    /// no new permutations are found. This is necessary for SYMMETRY sets like
    /// `Permutations(A) \cup Permutations(B)` where we need to try all combinations.
    fn compute_symmetry_perms(ctx: &EvalCtx, config: &Config) -> Vec<crate::value::FuncValue> {
        let symmetry_name = match &config.symmetry {
            Some(name) => name,
            None => return Vec::new(),
        };

        // Try to evaluate the symmetry operator
        // First, check if it's defined
        let def = match ctx.get_op(symmetry_name) {
            Some(d) => d.clone(),
            None => {
                eprintln!(
                    "Warning: SYMMETRY operator '{}' not found, symmetry reduction disabled",
                    symmetry_name
                );
                return Vec::new();
            }
        };

        // Evaluate the symmetry operator (should have no parameters)
        if !def.params.is_empty() {
            eprintln!(
                "Warning: SYMMETRY operator '{}' should have no parameters, symmetry reduction disabled",
                symmetry_name
            );
            return Vec::new();
        }

        // Evaluate in the context (no state variables needed for symmetry definition)
        match crate::eval::eval(ctx, &def.body) {
            Ok(value) => {
                // The result should be a set of functions
                if let Some(set) = value.as_set() {
                    let mut perms = Vec::new();
                    for v in set.iter() {
                        if let Some(func) = v.as_func() {
                            perms.push(func.clone());
                        } else {
                            eprintln!(
                                "Warning: SYMMETRY set contains non-function value, skipping"
                            );
                        }
                    }

                    let base_count = perms.len();
                    if base_count == 0 {
                        return Vec::new();
                    }

                    // Compute group closure by composing all pairs until no new permutations
                    // This is necessary for SYMMETRY = Permutations(A) \cup Permutations(B)
                    // where we need to try all |A|! * |B|! combinations
                    use std::collections::BTreeSet;
                    // Allow mutable_key_type: FuncValue's interior mutability is only for
                    // cached fingerprint which doesn't affect semantic ordering
                    #[allow(clippy::mutable_key_type)]
                    let mut seen: BTreeSet<crate::value::FuncValue> =
                        perms.iter().cloned().collect();
                    let mut changed = true;
                    let max_closure_size = 1000; // Safety limit to prevent infinite loops

                    while changed && seen.len() < max_closure_size {
                        changed = false;
                        let current: Vec<_> = seen.iter().cloned().collect();
                        for p1 in &current {
                            for p2 in &current {
                                let composed = p1.compose_perm(p2);
                                if !seen.contains(&composed) {
                                    seen.insert(composed);
                                    changed = true;
                                }
                            }
                        }
                    }

                    let closure: Vec<_> = seen.into_iter().collect();
                    let closure_count = closure.len();

                    if closure_count > base_count {
                        eprintln!(
                            "Symmetry reduction enabled: {} permutation(s) (closure of {} base)",
                            closure_count, base_count
                        );
                    } else {
                        eprintln!(
                            "Symmetry reduction enabled: {} permutation(s)",
                            closure_count
                        );
                    }

                    closure
                } else {
                    eprintln!(
                        "Warning: SYMMETRY operator '{}' did not return a set, symmetry reduction disabled",
                        symmetry_name
                    );
                    Vec::new()
                }
            }
            Err(e) => {
                eprintln!(
                    "Warning: Error evaluating SYMMETRY operator '{}': {:?}, symmetry reduction disabled",
                    symmetry_name, e
                );
                Vec::new()
            }
        }
    }

    /// Validate and cache the VIEW operator name from the configuration
    ///
    /// If VIEW is configured, validates that the operator exists and is a zero-parameter
    /// operator, then caches the name for use during fingerprinting.
    fn validate_view(&mut self) {
        let view_name = match &self.config.view {
            Some(name) => name.clone(),
            None => return,
        };

        // Look up the VIEW operator definition to validate it
        let def = match self.ctx.get_op(&view_name) {
            Some(d) => d.clone(),
            None => {
                eprintln!(
                    "Warning: VIEW operator '{}' not found, using full state fingerprints",
                    view_name
                );
                return;
            }
        };

        // VIEW should be a zero-argument operator
        if !def.params.is_empty() {
            eprintln!(
                "Warning: VIEW operator '{}' has {} parameters, expected 0. Using full state fingerprints",
                view_name, def.params.len()
            );
            return;
        }

        // Cache the view operator name
        self.cached_view_name = Some(view_name.clone());
        eprintln!("VIEW enabled: using '{}' for fingerprinting", view_name);
    }

    /// Set fairness constraints from a resolved SPECIFICATION formula
    ///
    /// These constraints will be conjoined with negated liveness properties
    /// during liveness checking.
    pub fn set_fairness(&mut self, fairness: Vec<FairnessConstraint>) {
        self.fairness = fairness;
    }

    /// Enable or disable per-action coverage statistics collection.
    ///
    /// When enabled, the model checker will enumerate each detected action separately and
    /// record per-action transition counts and "enabled in states" counts.
    pub fn set_collect_coverage(&mut self, collect: bool) {
        self.collect_coverage = collect;
    }

    fn update_coverage_totals(&mut self) {
        if let Some(ref mut coverage) = self.stats.coverage {
            coverage.total_states = self.stats.states_found;
            coverage.total_transitions = self.stats.transitions;
        }
    }

    /// Enable or disable deadlock checking
    pub fn set_deadlock_check(&mut self, check: bool) {
        self.check_deadlock = check;
    }

    /// Apply stuttering semantics from the resolved spec.
    ///
    /// When a TLA+ spec uses `[][Next]_vars` (square bracket form), stuttering is explicitly
    /// allowed - meaning the system can legally stay in the same state forever. In this case,
    /// reaching a state with no enabled actions is NOT a deadlock.
    ///
    /// When a spec uses `[]<<Next>>_vars` (angle bracket form), stuttering is forbidden -
    /// the system MUST make progress, so reaching a state with no enabled actions IS a deadlock.
    ///
    /// This method disables deadlock checking if:
    /// 1. The spec uses `[A]_v` form (stuttering_allowed = true), AND
    /// 2. CHECK_DEADLOCK was not explicitly set in the config
    ///
    /// Call this after creating the checker when using SPECIFICATION directive.
    pub fn apply_stuttering_semantics(&mut self, resolved: &ResolvedSpec) {
        if resolved.stuttering_allowed && !self.config.check_deadlock_explicit {
            self.check_deadlock = false;
        }
    }

    /// Set maximum number of states to explore
    ///
    /// When this limit is reached, model checking stops with `CheckResult::LimitReached`.
    /// This is useful for unbounded specifications that would otherwise run indefinitely.
    pub fn set_max_states(&mut self, limit: usize) {
        self.max_states = Some(limit);
    }

    /// Set maximum BFS depth to explore
    ///
    /// When this limit is reached, model checking stops with `CheckResult::LimitReached`.
    /// Depth 0 = initial states, depth 1 = first successors, etc.
    pub fn set_max_depth(&mut self, limit: usize) {
        self.max_depth = Some(limit);
    }

    /// Set a progress callback to receive periodic updates during model checking
    ///
    /// The callback is called approximately every `interval` states (default: 1000).
    /// This is useful for long-running model checks to show progress to users.
    pub fn set_progress_callback(&mut self, callback: ProgressCallback) {
        self.progress_callback = Some(callback);
    }

    /// Sync TLC config to the evaluation context for TLCGet("config") support
    ///
    /// The `mode` parameter specifies the exploration mode:
    /// - "bfs" for exhaustive model checking
    /// - "generate" for simulation/random behavior generation
    fn sync_tlc_config(&mut self, mode: &str) {
        use crate::eval::TlcConfig;
        let config = TlcConfig {
            mode: Arc::from(mode),
            depth: self.max_depth.map(|d| d as i64).unwrap_or(-1),
            deadlock: self.check_deadlock,
        };
        self.ctx.set_tlc_config(config);
    }

    /// Compute the fingerprint of a state, applying VIEW and symmetry reduction if configured
    ///
    /// Fingerprinting order of operations:
    /// 1. If VIEW is configured: evaluate the VIEW operator on the state and fingerprint the result
    /// 2. If symmetry permutations are configured: return the canonical fingerprint
    ///    (the smallest fingerprint among all symmetric states)
    /// 3. Otherwise: return the regular state fingerprint
    fn state_fingerprint(&mut self, state: &State) -> Fingerprint {
        // If VIEW is configured, evaluate it and fingerprint the result
        if let Some(ref view_name) = self.cached_view_name.clone() {
            // Save current environment scope
            let saved = self.ctx.save_scope();

            // Bind state variables to the environment
            for (name, value) in state.vars() {
                self.ctx.bind_mut(Arc::clone(name), value.clone());
            }

            // Evaluate the VIEW operator
            let view_result = self.ctx.eval_op(view_name);

            // Restore environment scope
            self.ctx.restore_scope(saved);

            // Compute fingerprint from the VIEW result
            match view_result {
                Ok(value) => {
                    // Compute fingerprint from the view value (wrap u64 in Fingerprint)
                    return Fingerprint(value_fingerprint(&value));
                }
                Err(e) => {
                    // Fall back to regular fingerprint on error
                    eprintln!("Warning: VIEW evaluation error: {:?}, using full state", e);
                }
            }
        }

        // For symmetry-based fingerprinting, use the cache
        if !self.symmetry_perms.is_empty() {
            let original_fp = state.fingerprint();
            // Check cache first
            if let Some(&canonical) = self.symmetry_fp_cache.get(&original_fp) {
                return canonical;
            }
            // Compute and cache
            let canonical = state.fingerprint_with_symmetry(&self.symmetry_perms);
            self.symmetry_fp_cache.insert(original_fp, canonical);
            return canonical;
        }

        // Regular fingerprint (no symmetry)
        state.fingerprint()
    }

    /// Compute fingerprint for an ArrayState.
    ///
    /// Fast path when no VIEW or symmetry is configured - uses ArrayState directly.
    /// Falls back to State-based fingerprint for VIEW/symmetry handling.
    fn array_state_fingerprint(&mut self, array_state: &mut ArrayState) -> Fingerprint {
        // Fast path: if fingerprint is already cached and no VIEW/symmetry, return it
        // This avoids registry access for states popped from queue
        if self.cached_view_name.is_none() && self.symmetry_perms.is_empty() {
            if let Some(fp) = array_state.cached_fingerprint() {
                return fp;
            }
        }

        let registry = self.ctx.var_registry().clone();

        // If VIEW is configured, fall back to State-based fingerprinting
        if self.cached_view_name.is_some() {
            let state = array_state.to_state(&registry);
            return self.state_fingerprint(&state);
        }

        // If symmetry is configured, fall back to State-based fingerprinting
        if !self.symmetry_perms.is_empty() {
            let state = array_state.to_state(&registry);
            return self.state_fingerprint(&state);
        }

        // Fast path: compute fingerprint directly from ArrayState
        array_state.fingerprint(&registry)
    }

    /// Set how often progress is reported (in number of states)
    ///
    /// Default is 1000 states. Setting to 0 disables progress reporting.
    pub fn set_progress_interval(&mut self, interval: usize) {
        self.progress_interval = interval;
    }

    /// Set whether to store full states for trace reconstruction
    ///
    /// When `store` is true (legacy mode):
    /// - Full states are stored in memory (42x more memory than fingerprint-only)
    /// - Faster trace reconstruction (no replay needed)
    /// - Required for liveness checking
    ///
    /// When `store` is false (default, #88):
    /// - Only fingerprints are stored, not full states
    /// - Significantly reduces memory usage for large state spaces
    /// - Counterexample traces reconstructed via temp trace file (unless disabled)
    ///
    /// Default is false (fingerprint-only for 42x memory reduction).
    pub fn set_store_states(&mut self, store: bool) {
        self.store_full_states = store;
        // Update liveness cache flag: only cache if we'll actually do liveness checking
        self.cache_successors_for_liveness =
            !self.config.properties.is_empty() && !skip_liveness() && store;
    }

    /// Returns whether full states are being stored
    pub fn stores_full_states(&self) -> bool {
        self.store_full_states
    }

    /// Set whether to auto-create a temp trace file for fingerprint-only mode
    ///
    /// When true (default): Creates a temporary trace file automatically if
    /// `store_full_states` is false and no explicit trace file is set.
    ///
    /// When false (--no-trace mode): No trace file is created, traces are
    /// completely unavailable for maximum memory efficiency.
    pub fn set_auto_create_trace_file(&mut self, auto_create: bool) {
        self.auto_create_trace_file = auto_create;
    }

    /// Set the fingerprint storage backend.
    ///
    /// This allows using memory-mapped storage for large state spaces that
    /// exceed available RAM. Must be called before `check()`.
    ///
    /// Only used when `store_full_states` is false (no-trace mode).
    /// When `store_full_states` is true, full states are stored in a HashMap
    /// regardless of this setting.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tla_check::{ModelChecker, FingerprintStorage, FingerprintSet};
    /// use std::sync::Arc;
    ///
    /// let mut checker = ModelChecker::new(&module, &config);
    /// checker.set_store_states(false); // Enable no-trace mode
    /// let storage = FingerprintStorage::mmap(10_000_000, None)?;
    /// checker.set_fingerprint_storage(Arc::new(storage) as Arc<dyn FingerprintSet>);
    /// let result = checker.check();
    /// ```
    pub fn set_fingerprint_storage(&mut self, storage: Arc<dyn FingerprintSet>) {
        self.seen_fps = storage;
    }

    /// Enable disk-based trace storage for large state space exploration.
    ///
    /// When enabled, the model checker writes (predecessor_loc, fingerprint) pairs
    /// to a disk file instead of keeping full states in memory. This significantly
    /// reduces memory usage while still enabling counterexample trace reconstruction.
    ///
    /// When a violation is found, the trace is reconstructed by:
    /// 1. Walking backward through the trace file to collect fingerprints
    /// 2. Replaying from the initial state, generating successors and matching
    ///    by fingerprint until the error state is reached
    ///
    /// # Arguments
    ///
    /// * `trace_file` - The trace file to use for storage
    ///
    /// # Notes
    ///
    /// - Trace file mode is incompatible with `store_full_states = true` (the two
    ///   approaches are mutually exclusive)
    /// - Trace reconstruction is slower than in-memory trace storage because states
    ///   must be regenerated from fingerprints
    /// - Liveness checking is disabled when using trace file mode (requires full states)
    pub fn set_trace_file(&mut self, trace_file: TraceFile) {
        self.trace_file = Some(trace_file);
        // Trace file mode implies we don't store full states in memory
        self.store_full_states = false;
        // No need to cache successors for liveness when full states aren't stored
        self.cache_successors_for_liveness = false;
    }

    /// Check if trace file mode is enabled.
    pub fn has_trace_file(&self) -> bool {
        self.trace_file.is_some()
    }

    /// Set the trace location storage for fingerprint-to-offset mapping.
    ///
    /// By default, trace locations are stored in memory. For large state spaces,
    /// use `TraceLocationsStorage::mmap()` to scale beyond available RAM.
    ///
    /// # Arguments
    ///
    /// * `storage` - The trace location storage to use
    pub fn set_trace_locations_storage(&mut self, storage: TraceLocationsStorage) {
        self.trace_locs = storage;
    }

    /// Enable checkpoint saving during model checking.
    ///
    /// Checkpoints are saved periodically to the specified directory, allowing
    /// interrupted model checking runs to be resumed.
    ///
    /// # Arguments
    ///
    /// * `dir` - Directory to save checkpoint files
    /// * `interval_secs` - Interval between checkpoints in seconds
    pub fn set_checkpoint(&mut self, dir: PathBuf, interval_secs: u64) {
        self.checkpoint_dir = Some(dir);
        self.checkpoint_interval = Duration::from_secs(interval_secs);
    }

    /// Set the spec and config paths for checkpoint metadata.
    ///
    /// These paths are stored in checkpoint metadata to help verify on resume
    /// that the checkpoint matches the current spec/config.
    pub fn set_checkpoint_paths(&mut self, spec_path: Option<String>, config_path: Option<String>) {
        self.checkpoint_spec_path = spec_path;
        self.checkpoint_config_path = config_path;
    }

    /// Check if checkpointing is enabled.
    pub fn is_checkpointing_enabled(&self) -> bool {
        self.checkpoint_dir.is_some()
    }

    /// Run the model checker
    pub fn check(&mut self) -> CheckResult {
        // Sync TLC config for TLCGet("config") support (must happen before ASSUME checking)
        self.sync_tlc_config("bfs");

        // Bind constants from config before checking
        if let Err(e) = bind_constants_from_config(&mut self.ctx, self.config) {
            return CheckResult::Error {
                error: CheckError::EvalError(e),
                stats: self.stats.clone(),
            };
        }

        // Compute symmetry permutations now that constants are bound
        if self.symmetry_perms.is_empty() && self.config.symmetry.is_some() {
            self.symmetry_perms = Self::compute_symmetry_perms(&self.ctx, self.config);
        }

        // Validate and cache VIEW operator name now that constants are bound
        if self.cached_view_name.is_none() && self.config.view.is_some() {
            self.validate_view();
        }

        // Check ASSUME statements before model checking
        // TLC checks all assumptions and stops if any evaluate to FALSE
        for (module_name, assume_expr) in &self.assumes {
            use crate::eval::eval;
            match eval(&self.ctx, assume_expr) {
                Ok(Value::Bool(true)) => {
                    // Assumption satisfied, continue
                }
                Ok(Value::Bool(false)) => {
                    // Assumption failed - format location like TLC
                    let location = format_span_location(&assume_expr.span, module_name);
                    return CheckResult::Error {
                        error: CheckError::AssumeFalse { location },
                        stats: self.stats.clone(),
                    };
                }
                Ok(_) => {
                    // Assumption evaluated to non-boolean (shouldn't happen for well-formed specs)
                    let location = format_span_location(&assume_expr.span, module_name);
                    return CheckResult::Error {
                        error: CheckError::AssumeFalse { location },
                        stats: self.stats.clone(),
                    };
                }
                Err(e) => {
                    // Evaluation error - report the error but with assumption context
                    return CheckResult::Error {
                        error: CheckError::EvalError(e),
                        stats: self.stats.clone(),
                    };
                }
            }
        }

        // Toolbox-generated "constant-expression evaluation" models often contain only
        // ASSUME statements (sometimes with Print/PrintT side effects) and do not provide
        // INIT/NEXT or SPECIFICATION. TLC treats these as successful runs after ASSUME
        // evaluation rather than hard errors.
        if self.config.init.is_none()
            && self.config.next.is_none()
            && self.config.specification.is_none()
            && self.vars.is_empty()
            && self.config.invariants.is_empty()
            && self.config.properties.is_empty()
            && !self.assumes.is_empty()
        {
            return CheckResult::Success(self.stats.clone());
        }

        // Validate config
        let init_name = match &self.config.init {
            Some(name) => name.clone(),
            None => {
                return CheckResult::Error {
                    error: CheckError::MissingInit,
                    stats: self.stats.clone(),
                }
            }
        };

        let next_name = match &self.config.next {
            Some(name) => name.clone(),
            None => {
                return CheckResult::Error {
                    error: CheckError::MissingNext,
                    stats: self.stats.clone(),
                }
            }
        };

        // Cache init/next names for trace reconstruction from fingerprints
        self.cached_init_name = Some(init_name.clone());
        self.cached_next_name = Some(next_name.clone());

        if self.vars.is_empty() {
            return CheckResult::Error {
                error: CheckError::NoVariables,
                stats: self.stats.clone(),
            };
        }

        // Validate invariants exist
        for inv_name in &self.config.invariants {
            if !self.ctx.has_op(inv_name) {
                return CheckResult::Error {
                    error: CheckError::MissingInvariant(inv_name.clone()),
                    stats: self.stats.clone(),
                };
            }
        }

        // Pre-compile invariants for efficient evaluation
        // This avoids AST traversal overhead per state (3.5s savings for bcastFolklore)
        //
        // Invariants containing ENABLED cannot be evaluated by the normal eval() path
        // because ENABLED requires knowledge of successor states. These invariants must
        // be routed to the liveness checker (checked as []Invariant).
        let registry = self.ctx.var_registry().clone();
        let empty_local_scope = LocalScope::new();
        self.enabled_invariants.clear();
        self.compiled_invariants = self
            .config
            .invariants
            .iter()
            .filter_map(|inv_name| {
                self.ctx.get_op(inv_name).and_then(|def| {
                    // Check if the invariant contains ENABLED operator
                    if contains_enabled(&def.body.node) {
                        // Route to liveness checker instead of normal invariant checking
                        self.enabled_invariants.push(inv_name.clone());
                        None
                    } else {
                        let compiled =
                            compile_guard(&self.ctx, &def.body, &registry, &empty_local_scope);
                        Some((inv_name.clone(), compiled))
                    }
                })
            })
            .collect();

        // Update cache_successors_for_liveness if we have ENABLED invariants
        // This must be done before BFS exploration since caching needs to be enabled from the start
        if !self.enabled_invariants.is_empty() && !skip_liveness() && self.store_full_states {
            self.cache_successors_for_liveness = true;
        }

        // Detect actions in the Next relation for coverage statistics
        if let Some(next_def) = self.op_defs.get(&next_name) {
            let actions = detect_actions(next_def);
            self.stats.detected_actions = actions.iter().map(|a| a.name.clone()).collect();

            if self.collect_coverage {
                let mut coverage = CoverageStats::new();
                for action in &actions {
                    coverage.register_action(action.name.clone());
                }
                self.coverage_actions = actions;
                self.stats.coverage = Some(coverage);
            } else {
                self.coverage_actions.clear();
                self.stats.coverage = None;
            }

        }

        // Auto-create temp trace file for fingerprint-only mode (#88)
        // This enables trace reconstruction while using 42x less memory than full-state storage.
        // Skip if user explicitly set a trace file, enabled full-state storage, or disabled auto-creation.
        if !self.store_full_states && self.auto_create_trace_file && self.trace_file.is_none() {
            match TraceFile::create_temp() {
                Ok(tf) => {
                    self.trace_file = Some(tf);
                }
                Err(_) => {
                    // Continue without trace file - traces will be unavailable
                    // but model checking will still work
                }
            }
        }

        // BFS exploration using ArrayState for O(1) state access
        let registry = self.ctx.var_registry().clone();

        // Track if we skipped states due to depth limit
        let mut depth_limit_reached = false;
        // Counter for progress reporting
        let mut states_since_progress = 0usize;
        // Start time for progress reporting
        let start_time = Instant::now();
        // Initialize checkpoint timing
        if self.checkpoint_dir.is_some() {
            self.last_checkpoint_time = Some(Instant::now());
        }

        if self.store_full_states {
            // --- Full-state mode ---
            // Try streaming enumeration first to avoid Vec<State> OrdMap overhead.
            // For large initial state sets (e.g., MCBakery ISpec with 655K states),
            // Vec<State> can consume 10+ GB from OrdMap allocations.
            let mut queue: VecDeque<Fingerprint> = VecDeque::new();

            let streaming_result = self.generate_initial_states_to_bulk(&init_name);
            let used_streaming = if let Ok(Some(bulk_storage)) = streaming_result {
                // Streaming successful! Process states from BulkStateStorage directly.
                // Filter by constraints and add to seen. Invariants are checked during
                // BFS to match original behavior (allows full trace reconstruction).
                let mut scratch = ArrayState::new(registry.len());
                let num_states = bulk_storage.len() as u32;

                // Filter by constraints and store states
                for idx in 0..num_states {
                    scratch.overwrite_from_slice(bulk_storage.get_state(idx));

                    // Check state constraints (CONSTRAINT directive)
                    if !self.check_state_constraints_array(&scratch) {
                        continue;
                    }

                    // Compute fingerprint for deduplication
                    let fp = self.array_state_fingerprint(&mut scratch);
                    if self.is_state_seen(fp) {
                        continue;
                    }

                    if debug_states() {
                        let state = scratch.to_state(&registry);
                        eprintln!("INIT STATE {}: {:?}", fp, state);
                    }

                    // Create ArrayState for storage (clone from scratch)
                    #[cfg(feature = "memory-stats")]
                    {
                        crate::value::memory_stats::inc_array_state();
                        crate::value::memory_stats::inc_array_state_bytes(scratch.len());
                    }
                    let mut arr = scratch.clone();
                    if self.cached_view_name.is_none() && self.symmetry_perms.is_empty() {
                        let _ = arr.fingerprint(&registry);
                    }
                    self.mark_state_seen_owned(fp, arr, None, 0);
                    queue.push_back(fp);
                }

                self.stats.initial_states = queue.len();

                // Note: Invariants are checked during BFS loop, matching original behavior.
                // Initial state invariant checks happen when states are popped from queue.

                true
            } else {
                false
            };

            // Fall back to Vec<State> path if streaming not available
            if !used_streaming {
                let initial_states = match self.generate_initial_states(&init_name) {
                    Ok(states) => states,
                    Err(e) => {
                        return CheckResult::Error {
                            error: e,
                            stats: self.stats.clone(),
                        }
                    }
                };

                // Filter initial states by state constraints (CONSTRAINT directive)
                let initial_states: Vec<State> = initial_states
                    .into_iter()
                    .filter(|state| self.check_state_constraints(state))
                    .collect();

                self.stats.initial_states = initial_states.len();

                // Check initial states satisfy invariants
                for state in &initial_states {
                    if let Some(violation) = self.check_invariants(state) {
                        let trace = Trace::from_states(vec![state.clone()]);
                        return CheckResult::InvariantViolation {
                            invariant: violation,
                            trace,
                            stats: self.stats.clone(),
                        };
                    }
                }

                // Mark initial states as seen with depth 0
                for state in &initial_states {
                    let fp = self.state_fingerprint(state);
                    if self.is_state_seen(fp) {
                        continue;
                    }
                    if debug_states() {
                        eprintln!("INIT STATE {}: {:?}", fp, state);
                    }
                    let mut arr = ArrayState::from_state(state, &registry);
                    if self.cached_view_name.is_none() && self.symmetry_perms.is_empty() {
                        let _ = arr.fingerprint(&registry);
                    }
                    self.mark_state_seen_owned(fp, arr, None, 0);
                    queue.push_back(fp);
                }

                // Explicitly drop Vec<State> to release OrdMap memory
                drop(initial_states);
            }

            // Initialize states_found with initial states count
            self.stats.states_found = self.states_count();

            while let Some(fp) = queue.pop_front() {
                let current_depth = *self.depths.get(&fp).unwrap_or(&0);

                // Progress reporting
                if self.progress_interval > 0 {
                    states_since_progress += 1;
                    if states_since_progress >= self.progress_interval {
                        states_since_progress = 0;
                        if let Some(ref callback) = self.progress_callback {
                            let elapsed_secs = start_time.elapsed().as_secs_f64();
                            let seen_count = self.states_count();
                            let states_per_sec = if elapsed_secs > 0.0 {
                                seen_count as f64 / elapsed_secs
                            } else {
                                0.0
                            };
                            let progress = Progress {
                                states_found: seen_count,
                                current_depth,
                                queue_size: queue.len(),
                                transitions: self.stats.transitions,
                                elapsed_secs,
                                states_per_sec,
                            };
                            callback(&progress);
                        }

                        // Check capacity warnings at progress intervals
                        self.check_and_warn_capacity();
                    }
                }

                // Take ownership of the current state so we can mutably update `self.seen`
                // while generating successors.
                let current_array = match self.seen.remove(&fp) {
                    Some(arr) => arr,
                    None => continue,
                };

                // Periodic checkpoint saving (requires State conversion)
                if let (Some(ref checkpoint_dir), Some(last_time)) =
                    (&self.checkpoint_dir, self.last_checkpoint_time)
                {
                    if last_time.elapsed() >= self.checkpoint_interval {
                        let state_frontier: VecDeque<State> = {
                            let mut frontier: VecDeque<State> = queue
                                .iter()
                                .filter_map(|q_fp| self.seen.get(q_fp))
                                .map(|arr| arr.to_state(&registry))
                                .collect();
                            frontier.push_front(current_array.to_state(&registry));
                            frontier
                        };
                        let checkpoint = self.create_checkpoint(
                            &state_frontier,
                            self.checkpoint_spec_path.as_deref(),
                            self.checkpoint_config_path.as_deref(),
                        );
                        if let Err(e) = checkpoint.save(checkpoint_dir) {
                            eprintln!("Warning: Failed to save checkpoint: {}", e);
                        } else {
                            eprintln!(
                                "Checkpoint saved: {} states, {} frontier",
                                self.states_count(),
                                state_frontier.len()
                            );
                        }
                        self.last_checkpoint_time = Some(Instant::now());
                    }
                }

                // Check state limit
                if let Some(max_states) = self.max_states {
                    if self.states_count() >= max_states {
                        self.stats.states_found = self.states_count();
                        self.update_coverage_totals();
                        // Restore current state before returning
                        self.seen.insert(fp, current_array);
                        // Print profiling stats before early return
                        print_enum_profile_stats();
                        print_symmetry_stats();
                        return CheckResult::LimitReached {
                            limit_type: LimitType::States,
                            stats: self.stats.clone(),
                        };
                    }
                }

                // Check depth limit before generating successors
                if let Some(max_depth) = self.max_depth {
                    if current_depth >= max_depth {
                        // Don't explore further from this state
                        depth_limit_reached = true;
                        self.seen.insert(fp, current_array);
                        continue;
                    }
                }

                // Try diff-based successor generation for memory efficiency.
                // Can only use diffs when: no VIEW, no symmetry, no liveness caching
                // (these require full state materialization for all successors).
                let use_diffs = self.cached_view_name.is_none()
                    && self.symmetry_perms.is_empty()
                    && !self.cache_successors_for_liveness;

                let succ_depth = current_depth + 1;

                if use_diffs {
                    // Fast path: diff-based successor generation
                    // Fingerprints computed incrementally, full states only materialized for unique successors
                    let diff_result = match self.generate_successors_as_diffs(&current_array) {
                        Ok(Some(result)) => Some(result),
                        Ok(None) => {
                            // Fall through to full-state path
                            None
                        }
                        Err(e) => {
                            self.update_coverage_totals();
                            self.seen.insert(fp, current_array);
                            return CheckResult::Error {
                                error: e,
                                stats: self.stats.clone(),
                            };
                        }
                    };

                    // Check if we got diffs or need to fall back
                    if let Some(succ_result) = diff_result {
                        let diffs = succ_result.successors;

                        let state_tlc_fp = if debug_successors_tlc_fp()
                            || debug_successors_filter_state_tlc_fp().is_some()
                        {
                            Some(tlc_fp_for_state_values(current_array.values()))
                        } else {
                            None
                        };
                        let debug_state = should_debug_successors_for_state(fp, state_tlc_fp);

                        // Debug: per-state successor counts for algorithm comparison
                        if debug_successors() && should_print_successor_debug_line(debug_state) {
                            if let Some(tlc_fp) = state_tlc_fp {
                                eprintln!(
                                    "STATE {:016x} tlc={:016x} depth={} -> {} successors (diff)",
                                    fp.0,
                                    tlc_fp,
                                    current_depth,
                                    diffs.len()
                                );
                            } else {
                                eprintln!(
                                    "STATE {:016x} depth={} -> {} successors (diff)",
                                    fp.0,
                                    current_depth,
                                    diffs.len()
                                );
                            }
                        }

                        if debug_state {
                            if let Some(tlc_fp) = state_tlc_fp {
                                eprintln!(
                                    "DEBUG SUCCESSORS state internal={:016x} tlc={:016x} depth={} successors={} had_raw={}",
                                    fp.0,
                                    tlc_fp,
                                    current_depth,
                                    diffs.len(),
                                    succ_result.had_raw_successors
                                );
                            } else {
                                eprintln!(
                                    "DEBUG SUCCESSORS state internal={:016x} depth={} successors={} had_raw={}",
                                    fp.0,
                                    current_depth,
                                    diffs.len(),
                                    succ_result.had_raw_successors
                                );
                            }

                            if debug_successors_dump_state() {
                                eprintln!(
                                    "DEBUG SUCCESSORS state value: {:?}",
                                    current_array.to_state(&registry)
                                );
                            }

                            let mut succ_debug = Vec::with_capacity(diffs.len());
                            for diff in &diffs {
                                let succ_state = diff.materialize(&current_array, &registry);
                                let succ_tlc_fp = tlc_fp_for_state_values(succ_state.values());
                                succ_debug.push((diff.fingerprint, succ_tlc_fp, diff.num_changes()));
                            }
                            succ_debug.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.0.cmp(&b.0.0)));
                            for (succ_fp, succ_tlc_fp, changes) in succ_debug {
                                eprintln!(
                                    "  succ internal={:016x} tlc={:016x} changes={}",
                                    succ_fp.0, succ_tlc_fp, changes
                                );
                            }
                        }

                        // Deadlock check
                        // TLC semantics: A state at the "edge" of the constrained state space (where all
                        // successors violate state constraints) is NOT considered a deadlock. Only states
                        // with truly no successors (before constraint filtering) are deadlocks.
                        if self.check_deadlock
                            && diffs.is_empty()
                            && !succ_result.had_raw_successors
                            && !self.is_terminal_state_array(&current_array)
                        {
                            self.stats.states_found = self.states_count();
                            self.update_coverage_totals();
                            self.seen.insert(fp, current_array);
                            let trace = self.reconstruct_trace(fp);
                            return CheckResult::Deadlock {
                                trace,
                                stats: self.stats.clone(),
                            };
                        }

                        // Update transition count
                        self.stats.transitions += diffs.len();
                        if let Some(ref mut coverage) = self.stats.coverage {
                            coverage.total_transitions = self.stats.transitions;
                        }

                        // Optimization: Materialize only new states before inserting current_array back.
                        // This avoids cloning current_array (which was previously needed because
                        // into_array_state borrows it while we also need to insert into self.seen).
                        // By batching materialization first, we can move current_array instead of cloning.
                        let new_successors: Vec<(Fingerprint, ArrayState)> = diffs
                            .into_iter()
                            .filter(|diff| !self.is_state_seen(diff.fingerprint))
                            .map(|diff| {
                                let succ_fp = diff.fingerprint;
                                let succ = diff.into_array_state(&current_array, &registry);
                                (succ_fp, succ)
                            })
                            .collect();

                        // Now we can put current state back without cloning
                        self.seen.insert(fp, current_array);

                        // Process materialized successors
                        for (succ_fp, succ) in new_successors {
                            if let Some(violation) = self.check_invariants_array(&succ) {
                                self.mark_state_seen_owned(succ_fp, succ, Some(fp), succ_depth);
                                self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                                self.stats.states_found = self.states_count();
                                self.update_coverage_totals();
                                let trace = self.reconstruct_trace(succ_fp);
                                return CheckResult::InvariantViolation {
                                    invariant: violation,
                                    trace,
                                    stats: self.stats.clone(),
                                };
                            }

                            if debug_states() {
                                let succ_state = succ.to_state(&registry);
                                eprintln!("NEW STATE {}: {:?}", succ_fp, succ_state);
                            }

                            self.mark_state_seen_owned(succ_fp, succ, Some(fp), succ_depth);
                            self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                            queue.push_back(succ_fp);

                            self.stats.states_found += 1;
                            self.stats.max_queue_depth =
                                self.stats.max_queue_depth.max(queue.len());
                        }
                        continue; // Next iteration of BFS
                    }
                }

                // Full-state path: generate all successors as ArrayStates
                // Used when VIEW, symmetry, liveness caching, or diff path unavailable
                let succ_result = match self.generate_successors_filtered_array(&current_array) {
                    Ok(result) => result,
                    Err(e) => {
                        self.update_coverage_totals();
                        self.seen.insert(fp, current_array);
                        return CheckResult::Error {
                            error: e,
                            stats: self.stats.clone(),
                        };
                    }
                };
                let valid_successors = succ_result.successors;
                let state_tlc_fp = if debug_successors_tlc_fp()
                    || debug_successors_filter_state_tlc_fp().is_some()
                {
                    Some(tlc_fp_for_state_values(current_array.values()))
                } else {
                    None
                };
                let debug_state = should_debug_successors_for_state(fp, state_tlc_fp);

                // Debug: per-state successor counts for algorithm comparison
                if debug_successors() && should_print_successor_debug_line(debug_state) {
                    if let Some(tlc_fp) = state_tlc_fp {
                        eprintln!(
                            "STATE {:016x} tlc={:016x} depth={} -> {} successors",
                            fp.0,
                            tlc_fp,
                            current_depth,
                            valid_successors.len()
                        );
                    } else {
                        eprintln!(
                            "STATE {:016x} depth={} -> {} successors",
                            fp.0,
                            current_depth,
                            valid_successors.len()
                        );
                    }
                }

                // Deadlock check
                // TLC semantics: A state at the "edge" of the constrained state space (where all
                // successors violate state constraints) is NOT considered a deadlock. Only states
                // with truly no successors (before constraint filtering) are deadlocks.
                if self.check_deadlock
                    && valid_successors.is_empty()
                    && !succ_result.had_raw_successors
                    && !self.is_terminal_state_array(&current_array)
                {
                    self.stats.states_found = self.states_count();
                    self.update_coverage_totals();
                    self.seen.insert(fp, current_array);
                    let trace = self.reconstruct_trace(fp);
                    return CheckResult::Deadlock {
                        trace,
                        stats: self.stats.clone(),
                    };
                }

                // Put current state back
                self.seen.insert(fp, current_array);

                // Pre-compute fingerprints for all successors
                let successors_with_fps: Vec<(ArrayState, Fingerprint)> = valid_successors
                    .into_iter()
                    .map(|mut arr| {
                        let fp_val = self.array_state_fingerprint(&mut arr);
                        (arr, fp_val)
                    })
                    .collect();

                if debug_state {
                    if let Some(tlc_fp) = state_tlc_fp {
                        eprintln!(
                            "DEBUG SUCCESSORS state internal={:016x} tlc={:016x} depth={} successors={} had_raw={}",
                            fp.0,
                            tlc_fp,
                            current_depth,
                            successors_with_fps.len(),
                            succ_result.had_raw_successors
                        );
                    } else {
                        eprintln!(
                            "DEBUG SUCCESSORS state internal={:016x} depth={} successors={} had_raw={}",
                            fp.0,
                            current_depth,
                            successors_with_fps.len(),
                            succ_result.had_raw_successors
                        );
                    }

                    if debug_successors_dump_state() {
                        if let Some(stored) = self.seen.get(&fp) {
                            eprintln!(
                                "DEBUG SUCCESSORS state value: {:?}",
                                stored.to_state(&registry)
                            );
                        } else {
                            eprintln!("DEBUG SUCCESSORS state value: <unavailable>");
                        }
                    }

                    let mut succ_debug = Vec::with_capacity(successors_with_fps.len());
                    for (succ_state, succ_fp) in &successors_with_fps {
                        let succ_tlc_fp = tlc_fp_for_state_values(succ_state.values());
                        succ_debug.push((*succ_fp, succ_tlc_fp));
                    }
                    succ_debug.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.0.cmp(&b.0.0)));
                    for (succ_fp, succ_tlc_fp) in succ_debug {
                        eprintln!("  succ internal={:016x} tlc={:016x}", succ_fp.0, succ_tlc_fp);
                    }
                }

                // Cache fingerprints for liveness checking (only if needed)
                if self.cache_successors_for_liveness {
                    let succ_fps: Vec<Fingerprint> =
                        successors_with_fps.iter().map(|(_, fp)| *fp).collect();
                    self.cached_successors.insert(fp, succ_fps);

                    if !self.symmetry_perms.is_empty() {
                        let succ_states: Vec<(Fingerprint, ArrayState)> = successors_with_fps
                            .iter()
                            .map(|(arr, succ_fp)| (*succ_fp, arr.clone()))
                            .collect();
                        self.cached_successor_states.insert(fp, succ_states);
                    }
                }

                self.stats.transitions += successors_with_fps.len();
                if let Some(ref mut coverage) = self.stats.coverage {
                    coverage.total_transitions = self.stats.transitions;
                }

                for (succ, succ_fp) in successors_with_fps {
                    if self.is_state_seen(succ_fp) {
                        continue;
                    }

                    if let Some(violation) = self.check_invariants_array(&succ) {
                        self.mark_state_seen_owned(succ_fp, succ, Some(fp), succ_depth);
                        self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                        self.stats.states_found = self.states_count();
                        self.update_coverage_totals();
                        let trace = self.reconstruct_trace(succ_fp);
                        return CheckResult::InvariantViolation {
                            invariant: violation,
                            trace,
                            stats: self.stats.clone(),
                        };
                    }

                    if debug_states() {
                        let succ_state = succ.to_state(&registry);
                        eprintln!("NEW STATE {}: {:?}", succ_fp, succ_state);
                    }

                    self.mark_state_seen_owned(succ_fp, succ, Some(fp), succ_depth);
                    self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                    queue.push_back(succ_fp);

                    self.stats.states_found += 1;
                    self.stats.max_queue_depth = self.stats.max_queue_depth.max(queue.len());
                }
            }
        } else {
            // --- No-trace mode ---
            // Memory-efficient path: try streaming enumeration directly to BulkStateStorage,
            // avoiding Vec<State> (OrdMap-based) intermediate representation entirely.
            //
            // For large initial state sets (e.g., MCBakery ISpec with 655K states), creating
            // a Vec<State> first causes massive memory usage from OrdMap allocations.
            // Streaming enumeration bypasses this by writing directly to contiguous storage.

            enum NoTraceQueueEntry {
                Bulk(BulkStateHandle),
                Owned(ArrayState),
            }

            // Scratch ArrayState used to process bulk-backed states without per-state allocation.
            let mut bulk_scratch = ArrayState::new(registry.len());

            // Try streaming enumeration first
            let (bulk_initial, mut queue) = match self.generate_initial_states_to_bulk(&init_name) {
                Ok(Some(storage)) => {
                    // Streaming successful! Now check constraints and invariants inline.
                    // This avoids creating any State (OrdMap) objects for passing states.
                    let mut queue: VecDeque<(NoTraceQueueEntry, usize)> = VecDeque::new();
                    queue.reserve(storage.len());

                    let num_states = storage.len() as u32;
                    for idx in 0..num_states {
                        // Load state into scratch buffer for constraint/invariant checking
                        bulk_scratch.overwrite_from_slice(storage.get_state(idx));

                        // Check state constraints (CONSTRAINT directive)
                        if !self.check_state_constraints_array(&bulk_scratch) {
                            continue; // Pruned by constraint, not an error
                        }

                        // Check fingerprint for deduplication
                        let fp = self.array_state_fingerprint(&mut bulk_scratch);
                        if self.is_state_seen(fp) {
                            continue;
                        }

                        // Check invariants BEFORE adding to queue
                        if let Some(violation) = self.check_invariants_array(&bulk_scratch) {
                            // Invariant violation! Convert to State for trace (only on error).
                            let violating_state = bulk_scratch.to_state(&registry);
                            let trace = Trace::from_states(vec![violating_state]);
                            return CheckResult::InvariantViolation {
                                invariant: violation,
                                trace,
                                stats: self.stats.clone(),
                            };
                        }

                        if debug_states() {
                            let state = bulk_scratch.to_state(&registry);
                            eprintln!("INIT STATE {}: {:?}", fp, state);
                        }

                        // State passed all checks - mark as seen and enqueue
                        self.mark_state_seen_fp_only(fp, None, 0);
                        queue.push_back((
                            NoTraceQueueEntry::Bulk(BulkStateHandle::with_fingerprint(idx, fp)),
                            0,
                        ));
                    }

                    self.stats.initial_states = queue.len();
                    (storage, queue)
                }
                Ok(None) | Err(_) => {
                    // Streaming not possible - fall back to Vec<State> path
                    let initial_states = match self.generate_initial_states(&init_name) {
                        Ok(states) => states,
                        Err(e) => {
                            return CheckResult::Error {
                                error: e,
                                stats: self.stats.clone(),
                            }
                        }
                    };

                    // Filter by constraints
                    let initial_states: Vec<State> = initial_states
                        .into_iter()
                        .filter(|state| self.check_state_constraints(state))
                        .collect();

                    // Check invariants
                    for state in &initial_states {
                        if let Some(violation) = self.check_invariants(state) {
                            let trace = Trace::from_states(vec![state.clone()]);
                            return CheckResult::InvariantViolation {
                                invariant: violation,
                                trace,
                                stats: self.stats.clone(),
                            };
                        }
                    }

                    // Convert to BulkStateStorage
                    let mut bulk_storage =
                        BulkStateStorage::new(registry.len(), initial_states.len());
                    let mut queue: VecDeque<(NoTraceQueueEntry, usize)> = VecDeque::new();
                    queue.reserve(initial_states.len());

                    for state in initial_states.into_iter() {
                        let fp = self.state_fingerprint(&state);
                        if self.is_state_seen(fp) {
                            continue;
                        }
                        if debug_states() {
                            eprintln!("INIT STATE {}: {:?}", fp, state);
                        }

                        let idx = bulk_storage.push_from_state(&state, &registry);
                        self.mark_state_seen_fp_only(fp, None, 0);
                        queue.push_back((
                            NoTraceQueueEntry::Bulk(BulkStateHandle::with_fingerprint(idx, fp)),
                            0,
                        ));
                    }

                    self.stats.initial_states = queue.len();
                    (bulk_storage, queue)
                }
            };

            // Initialize states_found with initial states count
            self.stats.states_found = self.states_count();

            // Profiling accumulators
            let do_profile = profile_enum();
            let mut prof_succ_gen_us: u64 = 0;
            let mut prof_fingerprint_us: u64 = 0;
            let mut prof_dedup_us: u64 = 0;
            let mut prof_invariant_us: u64 = 0;
            let mut prof_total_successors: u64 = 0;
            let mut prof_new_states: u64 = 0;

            while let Some((entry, current_depth)) = queue.pop_front() {
                let (bulk_handle, mut owned_current) = match entry {
                    NoTraceQueueEntry::Bulk(handle) => (Some(handle), None),
                    NoTraceQueueEntry::Owned(arr) => (None, Some(arr)),
                };

                let (fp, current_array): (Fingerprint, &mut ArrayState) =
                    if let Some(handle) = bulk_handle {
                        bulk_scratch.overwrite_from_slice(bulk_initial.get_state(handle.index));
                        let fp = handle
                            .fingerprint
                            .unwrap_or_else(|| self.array_state_fingerprint(&mut bulk_scratch));
                        (fp, &mut bulk_scratch)
                    } else {
                        let arr_ref = owned_current.as_mut().unwrap();
                        let fp = self.array_state_fingerprint(arr_ref);
                        (fp, arr_ref)
                    };

                // Progress reporting
                if self.progress_interval > 0 {
                    states_since_progress += 1;
                    if states_since_progress >= self.progress_interval {
                        states_since_progress = 0;
                        if let Some(ref callback) = self.progress_callback {
                            let elapsed_secs = start_time.elapsed().as_secs_f64();
                            let seen_count = self.states_count();
                            let states_per_sec = if elapsed_secs > 0.0 {
                                seen_count as f64 / elapsed_secs
                            } else {
                                0.0
                            };
                            let progress = Progress {
                                states_found: seen_count,
                                current_depth,
                                queue_size: queue.len(),
                                transitions: self.stats.transitions,
                                elapsed_secs,
                                states_per_sec,
                            };
                            callback(&progress);
                        }

                        // Check capacity warnings at progress intervals
                        self.check_and_warn_capacity();
                    }
                }

                // Periodic checkpoint saving (requires State conversion)
                if let (Some(ref checkpoint_dir), Some(last_time)) =
                    (&self.checkpoint_dir, self.last_checkpoint_time)
                {
                    if last_time.elapsed() >= self.checkpoint_interval {
                        // Convert ArrayState frontier to State for checkpoint
                        let state_frontier: VecDeque<State> = {
                            let mut frontier: VecDeque<State> = queue
                                .iter()
                                .map(|(entry, _)| match entry {
                                    NoTraceQueueEntry::Bulk(handle) => State::from_indexed(
                                        bulk_initial.get_state(handle.index),
                                        &registry,
                                    ),
                                    NoTraceQueueEntry::Owned(arr) => arr.to_state(&registry),
                                })
                                .collect();
                            frontier.push_front(current_array.to_state(&registry));
                            frontier
                        };
                        let checkpoint = self.create_checkpoint(
                            &state_frontier,
                            self.checkpoint_spec_path.as_deref(),
                            self.checkpoint_config_path.as_deref(),
                        );
                        if let Err(e) = checkpoint.save(checkpoint_dir) {
                            eprintln!("Warning: Failed to save checkpoint: {}", e);
                        } else {
                            eprintln!(
                                "Checkpoint saved: {} states, {} frontier",
                                self.states_count(),
                                state_frontier.len()
                            );
                        }
                        self.last_checkpoint_time = Some(Instant::now());
                    }
                }

                // Check state limit
                if let Some(max_states) = self.max_states {
                    if self.states_count() >= max_states {
                        self.stats.states_found = self.states_count();
                        self.update_coverage_totals();
                        // Print profiling stats before early return
                        print_enum_profile_stats();
                        return CheckResult::LimitReached {
                            limit_type: LimitType::States,
                            stats: self.stats.clone(),
                        };
                    }
                }

                // Check depth limit before generating successors
                if let Some(max_depth) = self.max_depth {
                    if current_depth >= max_depth {
                        // Don't explore further from this state
                        depth_limit_reached = true;
                        continue;
                    }
                }

                // Try diff-based successor generation for memory efficiency.
                // Can only use diffs when: no VIEW, no symmetry, no liveness caching.
                // The diff path avoids allocating full ArrayStates for duplicate successors,
                // which significantly reduces heap fragmentation for specs with high duplicate rates.
                let use_diffs = self.cached_view_name.is_none()
                    && self.symmetry_perms.is_empty()
                    && !self.cache_successors_for_liveness;

                let succ_depth = current_depth + 1;

                if use_diffs {
                    // Fast path: diff-based successor generation
                    let prof_t0 = if do_profile {
                        Instant::now()
                    } else {
                        start_time
                    };
                    let diff_result = match self.generate_successors_as_diffs(current_array) {
                        Ok(Some(result)) => Some(result),
                        Ok(None) => None, // Fall through to full-state path
                        Err(e) => {
                            self.update_coverage_totals();
                            return CheckResult::Error {
                                error: e,
                                stats: self.stats.clone(),
                            };
                        }
                    };
                    if do_profile {
                        prof_succ_gen_us += prof_t0.elapsed().as_micros() as u64;
                    }

                    // Check if we got diffs or need to fall back
                    if let Some(succ_result) = diff_result {
                        let diffs = succ_result.successors;

                        let state_tlc_fp = if debug_successors_tlc_fp()
                            || debug_successors_filter_state_tlc_fp().is_some()
                        {
                            Some(tlc_fp_for_state_values(current_array.values()))
                        } else {
                            None
                        };
                        let debug_state = should_debug_successors_for_state(fp, state_tlc_fp);

                        // Debug: per-state successor counts for algorithm comparison
                        if debug_successors() && should_print_successor_debug_line(debug_state) {
                            if let Some(tlc_fp) = state_tlc_fp {
                                eprintln!(
                                    "STATE {:016x} tlc={:016x} depth={} -> {} successors (diff/parallel)",
                                    fp.0,
                                    tlc_fp,
                                    current_depth,
                                    diffs.len()
                                );
                            } else {
                                eprintln!(
                                    "STATE {:016x} depth={} -> {} successors (diff/parallel)",
                                    fp.0,
                                    current_depth,
                                    diffs.len()
                                );
                            }
                        }

                        if debug_state {
                            if let Some(tlc_fp) = state_tlc_fp {
                                eprintln!(
                                    "DEBUG SUCCESSORS state internal={:016x} tlc={:016x} depth={} successors={} had_raw={}",
                                    fp.0,
                                    tlc_fp,
                                    current_depth,
                                    diffs.len(),
                                    succ_result.had_raw_successors
                                );
                            } else {
                                eprintln!(
                                    "DEBUG SUCCESSORS state internal={:016x} depth={} successors={} had_raw={}",
                                    fp.0,
                                    current_depth,
                                    diffs.len(),
                                    succ_result.had_raw_successors
                                );
                            }

                            if debug_successors_dump_state() {
                                eprintln!(
                                    "DEBUG SUCCESSORS state value: {:?}",
                                    current_array.to_state(&registry)
                                );
                            }

                            let mut succ_debug = Vec::with_capacity(diffs.len());
                            for diff in &diffs {
                                let succ_state = diff.materialize(current_array, &registry);
                                let succ_tlc_fp = tlc_fp_for_state_values(succ_state.values());
                                succ_debug.push((diff.fingerprint, succ_tlc_fp, diff.num_changes()));
                            }
                            succ_debug.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.0.cmp(&b.0.0)));
                            for (succ_fp, succ_tlc_fp, changes) in succ_debug {
                                eprintln!(
                                    "  succ internal={:016x} tlc={:016x} changes={}",
                                    succ_fp.0, succ_tlc_fp, changes
                                );
                            }
                        }

                        // Deadlock check
                        // TLC semantics: A state at the "edge" of the constrained state space (where all
                        // successors violate state constraints) is NOT considered a deadlock. Only states
                        // with truly no successors (before constraint filtering) are deadlocks.
                        if self.check_deadlock
                            && diffs.is_empty()
                            && !succ_result.had_raw_successors
                            && !self.is_terminal_state_array(current_array)
                        {
                            self.stats.states_found = self.states_count();
                            self.update_coverage_totals();
                            let trace = self.reconstruct_trace(fp);
                            return CheckResult::Deadlock {
                                trace,
                                stats: self.stats.clone(),
                            };
                        }

                        if do_profile {
                            prof_total_successors += diffs.len() as u64;
                        }
                        self.stats.transitions += diffs.len();
                        if let Some(ref mut coverage) = self.stats.coverage {
                            coverage.total_transitions = self.stats.transitions;
                        }

                        // Process diffs - only materialize for unique states
                        for diff in diffs {
                            let succ_fp = diff.fingerprint;

                            let prof_t2 = if do_profile {
                                Instant::now()
                            } else {
                                start_time
                            };
                            // Check if already seen BEFORE materializing
                            let is_new = !self.is_state_seen(succ_fp);
                            if do_profile {
                                prof_dedup_us += prof_t2.elapsed().as_micros() as u64;
                            }

                            if !is_new {
                                continue;
                            }

                            if do_profile {
                                prof_new_states += 1;
                            }

                            // New state - materialize and check invariants
                            let succ = diff.into_array_state(current_array, &registry);

                            let prof_t3 = if do_profile {
                                Instant::now()
                            } else {
                                start_time
                            };
                            let violation = self.check_invariants_array(&succ);
                            if do_profile {
                                prof_invariant_us += prof_t3.elapsed().as_micros() as u64;
                            }

                            if let Some(violation) = violation {
                                self.mark_state_seen(succ_fp, &succ, Some(fp), succ_depth);
                                self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                                self.stats.states_found = self.states_count();
                                self.update_coverage_totals();
                                let trace = self.reconstruct_trace(succ_fp);
                                return CheckResult::InvariantViolation {
                                    invariant: violation,
                                    trace,
                                    stats: self.stats.clone(),
                                };
                            }

                            if debug_states() {
                                let succ_state = succ.to_state(&registry);
                                eprintln!("NEW STATE {}: {:?}", succ_fp, succ_state);
                            }

                            // Store fingerprint only in no-trace mode
                            if self.store_full_states || self.trace_file.is_some() {
                                self.mark_state_seen(succ_fp, &succ, Some(fp), succ_depth);
                            } else {
                                self.seen_fps.insert(succ_fp);
                            }
                            self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                            queue.push_back((NoTraceQueueEntry::Owned(succ), succ_depth));

                            self.stats.states_found += 1;
                            self.stats.max_queue_depth =
                                self.stats.max_queue_depth.max(queue.len());
                        }
                        continue; // Next iteration of BFS
                    }
                }

                // Full-state path: generate all successors as ArrayStates
                // Used when VIEW, symmetry, liveness caching, or diff path unavailable
                let prof_t0 = if do_profile {
                    Instant::now()
                } else {
                    start_time
                };
                let succ_result = match self.generate_successors_filtered_array(current_array) {
                    Ok(result) => result,
                    Err(e) => {
                        self.update_coverage_totals();
                        return CheckResult::Error {
                            error: e,
                            stats: self.stats.clone(),
                        };
                    }
                };
                let valid_successors = succ_result.successors;
                if do_profile {
                    prof_succ_gen_us += prof_t0.elapsed().as_micros() as u64;
                }

                let state_tlc_fp = if debug_successors_tlc_fp()
                    || debug_successors_filter_state_tlc_fp().is_some()
                {
                    Some(tlc_fp_for_state_values(current_array.values()))
                } else {
                    None
                };
                let debug_state = should_debug_successors_for_state(fp, state_tlc_fp);

                // Debug: per-state successor counts for algorithm comparison
                if debug_successors() && should_print_successor_debug_line(debug_state) {
                    if let Some(tlc_fp) = state_tlc_fp {
                        eprintln!(
                            "STATE {:016x} tlc={:016x} depth={} -> {} successors (parallel)",
                            fp.0,
                            tlc_fp,
                            current_depth,
                            valid_successors.len()
                        );
                    } else {
                        eprintln!(
                            "STATE {:016x} depth={} -> {} successors (parallel)",
                            fp.0,
                            current_depth,
                            valid_successors.len()
                        );
                    }
                }

                // Deadlock check
                // TLC semantics: A state at the "edge" of the constrained state space (where all
                // successors violate state constraints) is NOT considered a deadlock. Only states
                // with truly no successors (before constraint filtering) are deadlocks.
                if self.check_deadlock
                    && valid_successors.is_empty()
                    && !succ_result.had_raw_successors
                    && !self.is_terminal_state_array(current_array)
                {
                    self.stats.states_found = self.states_count();
                    self.update_coverage_totals();
                    let trace = self.reconstruct_trace(fp);
                    return CheckResult::Deadlock {
                        trace,
                        stats: self.stats.clone(),
                    };
                }

                // Pre-compute fingerprints for all successors (single pass)
                // This avoids the double computation that was causing overhead
                let prof_t1 = if do_profile {
                    Instant::now()
                } else {
                    start_time
                };
                let successors_with_fps: Vec<(ArrayState, Fingerprint)> = valid_successors
                    .into_iter()
                    .map(|mut arr| {
                        let fp_val = self.array_state_fingerprint(&mut arr);
                        (arr, fp_val)
                    })
                    .collect();
                if do_profile {
                    prof_fingerprint_us += prof_t1.elapsed().as_micros() as u64;
                    prof_total_successors += successors_with_fps.len() as u64;
                }

                if debug_state {
                    if let Some(tlc_fp) = state_tlc_fp {
                        eprintln!(
                            "DEBUG SUCCESSORS state internal={:016x} tlc={:016x} depth={} successors={} had_raw={}",
                            fp.0,
                            tlc_fp,
                            current_depth,
                            successors_with_fps.len(),
                            succ_result.had_raw_successors
                        );
                    } else {
                        eprintln!(
                            "DEBUG SUCCESSORS state internal={:016x} depth={} successors={} had_raw={}",
                            fp.0,
                            current_depth,
                            successors_with_fps.len(),
                            succ_result.had_raw_successors
                        );
                    }

                    if debug_successors_dump_state() {
                        eprintln!(
                            "DEBUG SUCCESSORS state value: {:?}",
                            current_array.to_state(&registry)
                        );
                    }

                    let mut succ_debug = Vec::with_capacity(successors_with_fps.len());
                    for (succ_state, succ_fp) in &successors_with_fps {
                        let succ_tlc_fp = tlc_fp_for_state_values(succ_state.values());
                        succ_debug.push((*succ_fp, succ_tlc_fp));
                    }
                    succ_debug.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.0.cmp(&b.0.0)));
                    for (succ_fp, succ_tlc_fp) in succ_debug {
                        eprintln!("  succ internal={:016x} tlc={:016x}", succ_fp.0, succ_tlc_fp);
                    }
                }

                // Cache fingerprints for liveness checking (only if needed)
                if self.cache_successors_for_liveness {
                    let succ_fps: Vec<Fingerprint> =
                        successors_with_fps.iter().map(|(_, fp)| *fp).collect();
                    self.cached_successors.insert(fp, succ_fps);
                }

                self.stats.transitions += successors_with_fps.len();
                if let Some(ref mut coverage) = self.stats.coverage {
                    coverage.total_transitions = self.stats.transitions;
                }

                for (succ, succ_fp) in successors_with_fps {
                    let prof_t2 = if do_profile {
                        Instant::now()
                    } else {
                        start_time
                    };
                    let is_new = !self.is_state_seen(succ_fp);
                    if do_profile {
                        prof_dedup_us += prof_t2.elapsed().as_micros() as u64;
                    }
                    if is_new {
                        if do_profile {
                            prof_new_states += 1;
                        }
                        // New state - check invariants using ArrayState (fast path)
                        let prof_t3 = if do_profile {
                            Instant::now()
                        } else {
                            start_time
                        };
                        let violation = self.check_invariants_array(&succ);
                        if do_profile {
                            prof_invariant_us += prof_t3.elapsed().as_micros() as u64;
                        }
                        if let Some(violation) = violation {
                            // Store ArrayState directly (no conversion needed)
                            self.mark_state_seen(succ_fp, &succ, Some(fp), succ_depth);
                            self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                            self.stats.states_found = self.states_count();
                            self.update_coverage_totals();
                            let trace = self.reconstruct_trace(succ_fp);
                            return CheckResult::InvariantViolation {
                                invariant: violation,
                                trace,
                                stats: self.stats.clone(),
                            };
                        }

                        // Add to queue
                        if debug_states() {
                            let succ_state = succ.to_state(&registry);
                            eprintln!("NEW STATE {}: {:?}", succ_fp, succ_state);
                        }
                        // Store ArrayState directly when needed (no conversion to State)
                        if self.store_full_states || self.trace_file.is_some() {
                            self.mark_state_seen(succ_fp, &succ, Some(fp), succ_depth);
                        } else {
                            // Fast path: only record fingerprint, no ArrayState needed
                            self.seen_fps.insert(succ_fp);
                            // Note: depth is stored in queue tuple, no HashMap needed
                        }
                        self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                        queue.push_back((NoTraceQueueEntry::Owned(succ), succ_depth));

                        // Track state count directly to avoid calling states_count() which takes a lock
                        self.stats.states_found += 1;
                        self.stats.max_queue_depth = self.stats.max_queue_depth.max(queue.len());
                    }
                }
            }

            // Output profiling results
            if do_profile {
                let total_us = start_time.elapsed().as_micros() as u64;
                let other_us = total_us.saturating_sub(
                    prof_succ_gen_us + prof_fingerprint_us + prof_dedup_us + prof_invariant_us,
                );
                eprintln!("=== Enumeration Profile ===");
                eprintln!(
                    "  Successor gen:   {:>8.3}s ({:>5.1}%)",
                    prof_succ_gen_us as f64 / 1_000_000.0,
                    prof_succ_gen_us as f64 / total_us as f64 * 100.0
                );
                eprintln!(
                    "  Fingerprinting:  {:>8.3}s ({:>5.1}%)",
                    prof_fingerprint_us as f64 / 1_000_000.0,
                    prof_fingerprint_us as f64 / total_us as f64 * 100.0
                );
                eprintln!(
                    "  Dedup check:     {:>8.3}s ({:>5.1}%)",
                    prof_dedup_us as f64 / 1_000_000.0,
                    prof_dedup_us as f64 / total_us as f64 * 100.0
                );
                eprintln!(
                    "  Invariant check: {:>8.3}s ({:>5.1}%)",
                    prof_invariant_us as f64 / 1_000_000.0,
                    prof_invariant_us as f64 / total_us as f64 * 100.0
                );
                eprintln!(
                    "  Other:           {:>8.3}s ({:>5.1}%)",
                    other_us as f64 / 1_000_000.0,
                    other_us as f64 / total_us as f64 * 100.0
                );
                eprintln!("  ---");
                eprintln!("  Total:           {:>8.3}s", total_us as f64 / 1_000_000.0);
                eprintln!(
                    "  Total successors: {} ({:.0}/state)",
                    prof_total_successors,
                    prof_total_successors as f64 / prof_new_states as f64
                );
                eprintln!("  New states:       {}", prof_new_states);
            }
        }
        // Print detailed enumeration profile if enabled (has its own flag check)
        print_enum_profile_stats();

        self.stats.states_found = self.states_count();
        self.update_coverage_totals();

        // If we skipped states due to depth limit, report that
        if depth_limit_reached {
            self.update_coverage_totals();
            return CheckResult::LimitReached {
                limit_type: LimitType::Depth,
                stats: self.stats.clone(),
            };
        }

        self.update_coverage_totals();

        // Safety checking passed - now check liveness properties if any
        // Note: TLA2_SKIP_LIVENESS=1 can be set to skip liveness checking (for CI/benchmarks)
        // Note: In no-trace mode, liveness checking is disabled because it requires full states
        //
        // Also check invariants containing ENABLED - these must be checked via liveness
        // because ENABLED requires successor state information.
        let skip_liveness = skip_liveness() || !self.store_full_states;
        let has_liveness_work =
            !self.config.properties.is_empty() || !self.enabled_invariants.is_empty();
        if has_liveness_work && !skip_liveness {
            let liveness_prep_start = std::time::Instant::now();
            let registry = self.ctx.var_registry().clone();

            // Liveness checking needs `State` objects today, but our exploration stores `ArrayState`.
            //
            // Avoid converting the same `ArrayState` to `State` millions of times by building a
            // one-shot cache keyed by fingerprint. This keeps liveness performance dominated by
            // the actual tableau/graph algorithm instead of OrdMap construction.
            let state_cache_start = std::time::Instant::now();
            let mut state_cache: FxHashMap<Fingerprint, State> =
                FxHashMap::with_capacity_and_hasher(self.seen.len(), Default::default());
            for (fp, arr) in &self.seen {
                state_cache.insert(*fp, arr.to_state(&registry));
            }
            let state_cache_time = state_cache_start.elapsed();

            // Build a mapping from representative state fingerprints (regular) to their
            // canonical fingerprints (used for symmetry reduction).
            let fp_map_start = std::time::Instant::now();
            let mut state_fp_to_canon_fp: FxHashMap<Fingerprint, Fingerprint> =
                FxHashMap::with_capacity_and_hasher(state_cache.len(), Default::default());
            for (canon_fp, state) in &state_cache {
                state_fp_to_canon_fp.insert(state.fingerprint(), *canon_fp);
            }
            let state_fp_to_canon_fp_time = fp_map_start.elapsed();
            let state_fp_to_canon_fp = Arc::new(state_fp_to_canon_fp);

            // Under symmetry, liveness checking needs concrete successor states (not just
            // canonical fingerprints) to correctly evaluate ENABLED and action predicates.
            let succ_witness_start = std::time::Instant::now();
            let succ_witnesses: Option<Arc<SuccessorWitnessMap>> =
                if !self.symmetry_perms.is_empty() {
                    let mut out: SuccessorWitnessMap = FxHashMap::with_capacity_and_hasher(
                        self.cached_successor_states.len(),
                        Default::default(),
                    );
                    for (from_fp, succs) in &self.cached_successor_states {
                        let mut converted: Vec<(Fingerprint, State)> =
                            Vec::with_capacity(succs.len());
                        for (to_fp, arr) in succs {
                            converted.push((*to_fp, arr.to_state(&registry)));
                        }
                        out.insert(*from_fp, converted);
                    }
                    Some(Arc::new(out))
                } else {
                    None
                };
            let succ_witness_time = succ_witness_start.elapsed();

            // Collect initial states (those without parents)
            let init_states_start = std::time::Instant::now();
            let init_states: Vec<State> = state_cache
                .iter()
                .filter(|(fp, _)| !self.parents.contains_key(fp))
                .map(|(_, state)| state.clone())
                .collect();
            let init_states_time = init_states_start.elapsed();

            // Build successors map from cached fingerprints, resolving via the state cache.
            let succ_map_start = std::time::Instant::now();
            let mut state_successors: FxHashMap<Fingerprint, Vec<State>> =
                FxHashMap::with_capacity_and_hasher(
                    self.cached_successors.len(),
                    Default::default(),
                );
            for (fp, succ_fps) in &self.cached_successors {
                let mut succ_states: Vec<State> = Vec::with_capacity(succ_fps.len());
                for succ_fp in succ_fps {
                    if let Some(succ) = state_cache.get(succ_fp) {
                        succ_states.push(succ.clone());
                    }
                }
                state_successors.insert(*fp, succ_states);
            }
            let succ_map_time = succ_map_start.elapsed();
            let state_successors = Arc::new(state_successors);

            if std::env::var("LIVENESS_PROFILE").is_ok() {
                eprintln!("=== Liveness preparation ===");
                eprintln!(
                    "  state_cache build:   {:.3}s ({} states -> State)",
                    state_cache_time.as_secs_f64(),
                    state_cache.len()
                );
                eprintln!(
                    "  fp map build:        {:.3}s ({} state fps)",
                    state_fp_to_canon_fp_time.as_secs_f64(),
                    state_fp_to_canon_fp.len()
                );
                if !self.symmetry_perms.is_empty() {
                    eprintln!(
                        "  succ witness build:  {:.3}s ({} source states)",
                        succ_witness_time.as_secs_f64(),
                        succ_witnesses.as_ref().map(|m| m.len()).unwrap_or(0)
                    );
                }
                eprintln!(
                    "  init_states collect: {:.3}s ({} init states)",
                    init_states_time.as_secs_f64(),
                    init_states.len()
                );
                eprintln!(
                    "  succ_map build:      {:.3}s ({} entries, {} total succs)",
                    succ_map_time.as_secs_f64(),
                    state_successors.len(),
                    state_successors.values().map(|v| v.len()).sum::<usize>()
                );
                eprintln!(
                    "  Total prep time:     {:.3}s",
                    liveness_prep_start.elapsed().as_secs_f64()
                );
            }

            // Check each property
            for prop_name in &self.config.properties {
                let safety_temp_start = std::time::Instant::now();
                match self.check_safety_temporal_property(
                    prop_name,
                    &init_states,
                    state_successors.as_ref(),
                    &state_fp_to_canon_fp,
                    succ_witnesses.as_ref(),
                ) {
                    SafetyTemporalPropertyOutcome::NotApplicable => {}
                    SafetyTemporalPropertyOutcome::Satisfied => {
                        if std::env::var("LIVENESS_PROFILE").is_ok() {
                            eprintln!(
                                "  check_safety_temporal_property({}): {:.3}s (satisfied)",
                                prop_name,
                                safety_temp_start.elapsed().as_secs_f64()
                            );
                        }
                        continue;
                    }
                    SafetyTemporalPropertyOutcome::Violated(result) => return *result,
                }
                if std::env::var("LIVENESS_PROFILE").is_ok() {
                    eprintln!(
                        "  check_safety_temporal_property({}): {:.3}s",
                        prop_name,
                        safety_temp_start.elapsed().as_secs_f64()
                    );
                }

                if let Some(result) = self.check_liveness_property(
                    prop_name,
                    &init_states,
                    &state_successors,
                    &state_fp_to_canon_fp,
                    succ_witnesses.as_ref(),
                ) {
                    return result;
                }
            }

            // Check invariants containing ENABLED via liveness checker
            // These invariants are semantically []Invariant (must hold in all states)
            // but require the liveness checker because ENABLED needs successor state info.
            for inv_name in &self.enabled_invariants.clone() {
                if let Some(result) = self.check_enabled_invariant(
                    inv_name,
                    &init_states,
                    &state_successors,
                    &state_fp_to_canon_fp,
                    succ_witnesses.as_ref(),
                ) {
                    return result;
                }
            }
        }

        // Check for fingerprint storage errors before returning success
        if let Some(result) = self.check_fingerprint_storage_errors() {
            return result;
        }

        CheckResult::Success(self.stats.clone())
    }

    /// Resume model checking from a previously saved checkpoint.
    ///
    /// This method:
    /// 1. Loads checkpoint from the specified directory
    /// 2. Restores seen states, parents, and depths
    /// 3. Continues BFS exploration from the checkpoint's frontier
    ///
    /// # Arguments
    /// * `checkpoint_dir` - Directory containing the checkpoint files
    ///
    /// # Returns
    /// * `Ok(CheckResult)` - Result of continued model checking
    /// * `Err` - If checkpoint loading fails
    pub fn check_with_resume<P: AsRef<std::path::Path>>(
        &mut self,
        checkpoint_dir: P,
    ) -> Result<CheckResult, std::io::Error> {
        use crate::checkpoint::Checkpoint;

        // Load the checkpoint
        let checkpoint = Checkpoint::load(checkpoint_dir)?;

        eprintln!(
            "Resuming from checkpoint: {} states, {} frontier",
            checkpoint.fingerprints.len(),
            checkpoint.frontier.len()
        );

        // Bind constants from config before checking
        if let Err(e) = bind_constants_from_config(&mut self.ctx, self.config) {
            return Ok(CheckResult::Error {
                error: CheckError::EvalError(e),
                stats: self.stats.clone(),
            });
        }

        // Compute symmetry permutations
        if self.symmetry_perms.is_empty() && self.config.symmetry.is_some() {
            self.symmetry_perms = Self::compute_symmetry_perms(&self.ctx, self.config);
        }

        // Validate VIEW operator
        if self.cached_view_name.is_none() && self.config.view.is_some() {
            self.validate_view();
        }

        // Validate config
        let next_name = match &self.config.next {
            Some(name) => name.clone(),
            None => {
                return Ok(CheckResult::Error {
                    error: CheckError::MissingNext,
                    stats: self.stats.clone(),
                })
            }
        };

        // Cache init/next names for trace reconstruction
        self.cached_init_name = self.config.init.clone();
        self.cached_next_name = Some(next_name.clone());

        if self.vars.is_empty() {
            return Ok(CheckResult::Error {
                error: CheckError::NoVariables,
                stats: self.stats.clone(),
            });
        }

        // Validate invariants
        for inv_name in &self.config.invariants {
            if !self.ctx.has_op(inv_name) {
                return Ok(CheckResult::Error {
                    error: CheckError::MissingInvariant(inv_name.clone()),
                    stats: self.stats.clone(),
                });
            }
        }

        // Restore state from checkpoint
        let frontier = self.restore_from_checkpoint(checkpoint);

        // Continue BFS from the checkpoint frontier
        let mut queue: VecDeque<State> = frontier;

        // Track depth limit
        let mut depth_limit_reached = false;
        let mut states_since_progress = 0usize;
        let start_time = Instant::now();

        // Initialize checkpoint timing
        if self.checkpoint_dir.is_some() {
            self.last_checkpoint_time = Some(Instant::now());
        }

        while let Some(state) = queue.pop_front() {
            let fp = self.state_fingerprint(&state);
            let current_depth = *self.depths.get(&fp).unwrap_or(&0);

            // Progress reporting
            if self.progress_interval > 0 {
                states_since_progress += 1;
                if states_since_progress >= self.progress_interval {
                    states_since_progress = 0;
                    if let Some(ref callback) = self.progress_callback {
                        let elapsed_secs = start_time.elapsed().as_secs_f64();
                        let seen_count = self.states_count();
                        let states_per_sec = if elapsed_secs > 0.0 {
                            seen_count as f64 / elapsed_secs
                        } else {
                            0.0
                        };
                        let progress = Progress {
                            states_found: seen_count,
                            current_depth,
                            queue_size: queue.len(),
                            transitions: self.stats.transitions,
                            elapsed_secs,
                            states_per_sec,
                        };
                        callback(&progress);
                    }

                    // Check capacity warnings at progress intervals
                    self.check_and_warn_capacity();
                }
            }

            // Periodic checkpoint saving
            if let (Some(ref checkpoint_dir), Some(last_time)) =
                (&self.checkpoint_dir, self.last_checkpoint_time)
            {
                if last_time.elapsed() >= self.checkpoint_interval {
                    // Include the current state being processed in the checkpoint frontier so a
                    // resume will not skip work if we stop between popping and exploring.
                    let mut frontier = queue.clone();
                    frontier.push_front(state.clone());
                    let checkpoint = self.create_checkpoint(
                        &frontier,
                        self.checkpoint_spec_path.as_deref(),
                        self.checkpoint_config_path.as_deref(),
                    );
                    if let Err(e) = checkpoint.save(checkpoint_dir) {
                        eprintln!("Warning: Failed to save checkpoint: {}", e);
                    } else {
                        eprintln!(
                            "Checkpoint saved: {} states, {} frontier",
                            self.states_count(),
                            frontier.len()
                        );
                    }
                    self.last_checkpoint_time = Some(Instant::now());
                }
            }

            // Check state limit
            if let Some(max_states) = self.max_states {
                if self.states_count() >= max_states {
                    self.stats.states_found = self.states_count();
                    self.update_coverage_totals();
                    // Print profiling stats before early return
                    print_enum_profile_stats();
                    return Ok(CheckResult::LimitReached {
                        limit_type: LimitType::States,
                        stats: self.stats.clone(),
                    });
                }
            }

            // Check depth limit
            if let Some(max_depth) = self.max_depth {
                if current_depth >= max_depth {
                    depth_limit_reached = true;
                    continue;
                }
            }

            // Generate successors
            let succ_result = match self.generate_successors_filtered(&next_name, &state) {
                Ok(result) => result,
                Err(e) => {
                    self.update_coverage_totals();
                    return Ok(CheckResult::Error {
                        error: e,
                        stats: self.stats.clone(),
                    });
                }
            };
            let valid_successors = succ_result.successors;

            // Deadlock check
            // Skip deadlock check if this is a terminal state
            // TLC semantics: A state at the "edge" of the constrained state space (where all
            // successors violate state constraints) is NOT considered a deadlock. Only states
            // with truly no successors (before constraint filtering) are deadlocks.
            if self.check_deadlock
                && valid_successors.is_empty()
                && !succ_result.had_raw_successors
                && !self.is_terminal_state(&state)
            {
                self.stats.states_found = self.states_count();
                self.update_coverage_totals();
                let trace = self.reconstruct_trace(fp);
                return Ok(CheckResult::Deadlock {
                    trace,
                    stats: self.stats.clone(),
                });
            }

            // Get registry for ArrayState conversion (needed for liveness witness caching).
            let registry = self.ctx.var_registry().clone();

            // Cache successors for liveness (only if needed)
            if self.cache_successors_for_liveness {
                let succ_fps: Vec<Fingerprint> = valid_successors
                    .iter()
                    .map(|s| self.state_fingerprint(s))
                    .collect();
                self.cached_successors.insert(fp, succ_fps);

                if !self.symmetry_perms.is_empty() {
                    let succ_states: Vec<(Fingerprint, ArrayState)> = valid_successors
                        .iter()
                        .map(|s| {
                            (
                                self.state_fingerprint(s),
                                ArrayState::from_state(s, &registry),
                            )
                        })
                        .collect();
                    self.cached_successor_states.insert(fp, succ_states);
                }
            }

            self.stats.transitions += valid_successors.len();
            let succ_depth = current_depth + 1;

            for succ in valid_successors {
                let succ_fp = self.state_fingerprint(&succ);

                if !self.is_state_seen(succ_fp) {
                    // Convert to ArrayState for storage
                    let succ_array = ArrayState::from_state(&succ, &registry);

                    // Check invariants
                    if let Some(violation) = self.check_invariants(&succ) {
                        self.mark_state_seen(succ_fp, &succ_array, Some(fp), succ_depth);
                        self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                        self.stats.states_found = self.states_count();
                        self.update_coverage_totals();
                        let trace = self.reconstruct_trace(succ_fp);
                        return Ok(CheckResult::InvariantViolation {
                            invariant: violation,
                            trace,
                            stats: self.stats.clone(),
                        });
                    }

                    // Add to queue
                    self.mark_state_seen(succ_fp, &succ_array, Some(fp), succ_depth);
                    self.stats.max_depth = self.stats.max_depth.max(succ_depth);
                    queue.push_back(succ);

                    self.stats.states_found = self.states_count();
                    self.stats.max_queue_depth = self.stats.max_queue_depth.max(queue.len());
                }
            }
        }

        self.stats.states_found = self.states_count();
        self.update_coverage_totals();

        if depth_limit_reached {
            return Ok(CheckResult::LimitReached {
                limit_type: LimitType::Depth,
                stats: self.stats.clone(),
            });
        }

        // Note: Liveness checking after resume would require initial states
        // For now, we skip liveness checking on resumed runs
        // Full liveness support would need to track initial states in checkpoint

        // Check for fingerprint storage errors before returning success
        if let Some(result) = self.check_fingerprint_storage_errors() {
            return Ok(result);
        }

        Ok(CheckResult::Success(self.stats.clone()))
    }

    /// Generate initial states by finding all states satisfying Init
    fn generate_initial_states(&mut self, init_name: &str) -> Result<Vec<State>, CheckError> {
        self.solve_predicate_for_states(init_name)
    }

    /// Generate initial states directly to BulkStateStorage (memory-efficient for no-trace mode).
    ///
    /// This bypasses Vec<State> creation entirely, avoiding OrdMap allocations.
    /// Returns None if streaming enumeration is not possible (caller should fall back to Vec<State>).
    ///
    /// Used by no-trace mode to stream initial states directly to contiguous storage,
    /// with constraint and invariant checking done inline on the BulkStateStorage entries.
    fn generate_initial_states_to_bulk(
        &mut self,
        init_name: &str,
    ) -> Result<Option<BulkStateStorage>, CheckError> {
        self.solve_predicate_for_states_to_bulk(init_name)
    }

    /// Generate successor states from a given state via Next relation
    fn generate_successors(
        &mut self,
        next_name: &str,
        state: &State,
    ) -> Result<Vec<State>, CheckError> {
        // Similar to initial states, we need to find all states s' such that
        // Next(s, s') is true, where s is the current state.

        // For simple Next relations, we can enumerate possible next states
        // by trying all possible variable updates.

        // Simplified: try to find states by evaluating Next with current state bound
        let successors;

        // Bind current state variables to unprimed names
        let saved = self.ctx.save_scope();
        for (name, value) in state.vars() {
            self.ctx.bind_mut(Arc::clone(name), value.clone());
        }

        // Try to solve for primed variables
        // This is a simplified placeholder - real implementation needs
        // proper constraint solving or enumeration
        let next_states = self.solve_next_relation(next_name, state)?;

        self.ctx.restore_scope(saved);

        successors = next_states;

        Ok(successors)
    }

    /// Generate successor states from a given state via Next relation, filtered by state constraints.
    ///
    /// When coverage collection is enabled, this enumerates each detected action separately so we
    /// can attribute transitions to actions.
    fn generate_successors_filtered(
        &mut self,
        next_name: &str,
        state: &State,
    ) -> Result<SuccessorResult<Vec<State>>, CheckError> {
        if !self.collect_coverage || self.coverage_actions.is_empty() {
            let successors = self.generate_successors(next_name, state)?;
            let had_raw_successors = !successors.is_empty();
            let mut valid = Vec::new();
            for succ in successors {
                if self.check_state_constraints(&succ)
                    && self.check_action_constraints(state, &succ)
                {
                    valid.push(succ);
                }
            }
            return Ok(SuccessorResult {
                successors: valid,
                had_raw_successors,
            });
        }

        let actions = self.coverage_actions.clone();
        let (template_name, template_params, template_local) = {
            let next_def = self.op_defs.get(next_name).ok_or(CheckError::MissingNext)?;
            (
                next_def.name.clone(),
                next_def.params.clone(),
                next_def.local,
            )
        };

        let saved = self.ctx.save_scope();
        let mut all_valid_successors = Vec::new();
        let mut had_any_raw_successors = false;

        // Reuse the same OperatorDef, swapping the body for each action.
        let mut action_def = OperatorDef {
            name: template_name,
            params: template_params,
            body: actions[0].expr.clone(),
            local: template_local,
        };

        for action in actions {
            action_def.body = action.expr;
            let successors =
                match enumerate_successors(&mut self.ctx, &action_def, state, &self.vars) {
                    Ok(succ) => succ,
                    Err(e) => {
                        self.ctx.restore_scope(saved);
                        return Err(CheckError::EvalError(e));
                    }
                };

            if !successors.is_empty() {
                had_any_raw_successors = true;
            }

            let mut valid = Vec::new();
            for succ in successors {
                if self.check_state_constraints(&succ)
                    && self.check_action_constraints(state, &succ)
                {
                    valid.push(succ);
                }
            }

            if let Some(ref mut coverage) = self.stats.coverage {
                coverage.record_action(&action.name, valid.len());
            }

            all_valid_successors.extend(valid);
        }

        self.ctx.restore_scope(saved);
        Ok(SuccessorResult {
            successors: all_valid_successors,
            had_raw_successors: had_any_raw_successors,
        })
    }

    /// Check all invariants for a state, returning the first violated invariant
    fn check_invariants(&mut self, state: &State) -> Option<String> {
        let saved = self.ctx.save_scope();
        for (name, value) in state.vars() {
            self.ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let result = self
            .config
            .invariants
            .iter()
            .find(|inv_name| match self.ctx.eval_op(inv_name) {
                Ok(Value::Bool(true)) => false,
                Ok(Value::Bool(false)) => {
                    if debug_invariants() {
                        eprintln!("[invariant] {} = FALSE", inv_name);
                    }
                    true
                }
                Ok(other) => {
                    if debug_invariants() {
                        eprintln!("[invariant] {} = non-boolean ({:?})", inv_name, other);
                    }
                    true
                }
                Err(e) => {
                    if debug_invariants() {
                        eprintln!("[invariant] {} = error ({}) {:?}", inv_name, e, e);
                    }
                    true
                }
            })
            .cloned();

        self.ctx.restore_scope(saved);
        result
    }

    /// Check all invariants for an ArrayState, returning the first violated invariant
    ///
    /// This is the fast path that uses pre-compiled invariants to avoid AST traversal.
    fn check_invariants_array(&mut self, array_state: &ArrayState) -> Option<String> {
        // Bind state variables for invariant evaluation
        // (needed for Fallback paths in compiled invariants and for direct eval_op)
        let prev_state_env = self.ctx.bind_state_array(array_state.values());

        // Use pre-compiled invariants if available (fast path)
        let result = if !self.compiled_invariants.is_empty() {
            let mut violation = None;
            for (inv_name, compiled_guard) in &self.compiled_invariants {
                match compiled_guard.eval_with_array(&mut self.ctx, array_state) {
                    Ok(true) => {} // Invariant satisfied, continue
                    Ok(false) => {
                        if debug_invariants() {
                            eprintln!("[invariant] {} = FALSE", inv_name);
                        }
                        violation = Some(inv_name.clone());
                        break;
                    }
                    Err(e) => {
                        if debug_invariants() {
                            eprintln!("[invariant] {} = error ({}) {:?}", inv_name, e, e);
                        }
                        violation = Some(inv_name.clone());
                        break;
                    }
                }
            }
            violation
        } else {
            // Fallback to eval_op for compatibility (e.g., during simulation)
            self.config
                .invariants
                .iter()
                .find(|inv_name| {
                    match self.ctx.eval_op(inv_name) {
                        Ok(Value::Bool(true)) => false,
                        Ok(Value::Bool(false)) => true,
                        _ => true, // Error or non-boolean = violation
                    }
                })
                .cloned()
        };

        if debug_invariants() {
            if let Some(ref inv_name) = result {
                match self.ctx.eval_op(inv_name) {
                    Ok(Value::Bool(true)) => eprintln!("[invariant] {} = TRUE", inv_name),
                    Ok(Value::Bool(false)) => eprintln!("[invariant] {} = FALSE", inv_name),
                    Ok(other) => eprintln!("[invariant] {} = non-boolean ({:?})", inv_name, other),
                    Err(e) => eprintln!("[invariant] {} = error ({}) {:?}", inv_name, e, e),
                }
            }
        }

        self.ctx.restore_state_env(prev_state_env);
        result
    }

    /// Check all state constraints for an ArrayState, returning true if ALL are satisfied
    fn check_state_constraints_array(&mut self, array_state: &ArrayState) -> bool {
        if self.config.constraints.is_empty() {
            return true;
        }

        let prev_state_env = self.ctx.bind_state_array(array_state.values());

        let result = self.config.constraints.iter().all(|constraint_name| {
            matches!(self.ctx.eval_op(constraint_name), Ok(Value::Bool(true)))
        });

        self.ctx.restore_state_env(prev_state_env);
        result
    }

    /// Check all state constraints for a state, returning true if ALL are satisfied.
    ///
    /// State constraints (CONSTRAINT in config) limit the state space by excluding
    /// states that don't satisfy them. Unlike invariants, violating a constraint
    /// doesn't produce an error - the state is simply not explored.
    fn check_state_constraints(&mut self, state: &State) -> bool {
        // If no constraints, all states pass
        if self.config.constraints.is_empty() {
            return true;
        }

        let saved = self.ctx.save_scope();
        for (name, value) in state.vars() {
            self.ctx.bind_mut(Arc::clone(name), value.clone());
        }

        // All constraints must evaluate to true
        let result = self.config.constraints.iter().all(|constraint_name| {
            matches!(self.ctx.eval_op(constraint_name), Ok(Value::Bool(true)))
        });

        self.ctx.restore_scope(saved);
        result
    }

    /// Check all action constraints for a transition, returning true if ALL are satisfied.
    ///
    /// Action constraints (ACTION_CONSTRAINT in config) limit transitions by excluding
    /// those that don't satisfy them. Unlike state constraints (which only see the target
    /// state), action constraints can reference both primed (next-state) and unprimed
    /// (current-state) variables.
    fn check_action_constraints(&mut self, current: &State, next: &State) -> bool {
        // If no action constraints, all transitions pass
        if self.config.action_constraints.is_empty() {
            return true;
        }

        let saved = self.ctx.save_scope();
        let saved_next = self.ctx.next_state.clone();

        // Bind current state variables (unprimed)
        for (name, value) in current.vars() {
            self.ctx.bind_mut(Arc::clone(name), value.clone());
        }

        // Set up next state for primed variable access
        let mut next_env = Env::new();
        for (name, value) in next.vars() {
            next_env.insert(Arc::clone(name), value.clone());
        }
        self.ctx.next_state = Some(std::sync::Arc::new(next_env));

        // All action constraints must evaluate to true
        let result = self
            .config
            .action_constraints
            .iter()
            .all(|constraint_name| {
                matches!(self.ctx.eval_op(constraint_name), Ok(Value::Bool(true)))
            });

        self.ctx.next_state = saved_next;
        self.ctx.restore_scope(saved);
        result
    }

    /// Check if a state is a terminal state (should not be considered a deadlock)
    ///
    /// Terminal states are intentional end points of a spec, such as "SAT" or "UNSAT"
    /// in a SAT solver spec. They should not be reported as deadlocks.
    fn is_terminal_state(&mut self, state: &State) -> bool {
        let Some(terminal) = &self.config.terminal else {
            return false;
        };

        match terminal {
            TerminalSpec::Predicates(preds) => {
                // Check if any predicate matches the current state
                for (var_name, expected_val_str) in preds {
                    if let Some(actual_val) = state.get(var_name) {
                        // Parse the expected value string and compare
                        let expected =
                            crate::constants::parse_constant_value(expected_val_str.as_str());
                        if let Ok(expected_val) = expected {
                            if actual_val == &expected_val {
                                return true;
                            }
                        }
                    }
                }
                false
            }
            TerminalSpec::Operator(op_name) => {
                // Bind state variables and evaluate the operator
                let saved = self.ctx.save_scope();
                for (name, value) in state.vars() {
                    self.ctx.bind_mut(Arc::clone(name), value.clone());
                }

                let result = matches!(self.ctx.eval_op(op_name), Ok(Value::Bool(true)));

                self.ctx.restore_scope(saved);
                result
            }
        }
    }

    /// Check if an ArrayState is a terminal state
    fn is_terminal_state_array(&mut self, array_state: &ArrayState) -> bool {
        let Some(terminal) = &self.config.terminal else {
            return false;
        };

        match terminal {
            TerminalSpec::Predicates(preds) => {
                // Get variable names from registry
                let registry = self.ctx.var_registry();
                for (var_name, expected_val_str) in preds {
                    if let Some(idx) = registry.get(var_name) {
                        let actual_val = &array_state.values()[idx.as_usize()];
                        // Parse the expected value string and compare
                        let expected =
                            crate::constants::parse_constant_value(expected_val_str.as_str());
                        if let Ok(expected_val) = expected {
                            if actual_val == &expected_val {
                                return true;
                            }
                        }
                    }
                }
                false
            }
            TerminalSpec::Operator(op_name) => {
                // Bind state variables and evaluate the operator
                let prev_state_env = self.ctx.bind_state_array(array_state.values());

                let result = matches!(self.ctx.eval_op(op_name), Ok(Value::Bool(true)));

                self.ctx.restore_state_env(prev_state_env);
                result
            }
        }
    }

    /// Solve a predicate to find satisfying states
    fn solve_predicate_for_states(&mut self, pred_name: &str) -> Result<Vec<State>, CheckError> {
        // Resolve operator replacements (e.g., `Init <- MCInit` in config)
        let resolved_name = self.ctx.resolve_op_name(pred_name);

        // Get the operator definition for the predicate
        let def = self
            .op_defs
            .get(resolved_name)
            .ok_or(CheckError::MissingInit)?;

        // Try to extract constraints directly from the Init predicate.
        //
        // If this fails (unsupported expressions or missing per-variable constraints), we fall
        // back to enumerating states from a type constraint (usually `TypeOK`) and then filter
        // them by evaluating the full predicate.
        let direct_hint = if let Some(branches) =
            extract_init_constraints(&self.ctx, &def.body, &self.vars)
        {
            let unconstrained = find_unconstrained_vars(&self.vars, &branches);
            if unconstrained.is_empty() {
                return match enumerate_states_from_constraint_branches(
                    Some(&self.ctx),
                    &self.vars,
                    &branches,
                ) {
                    Some(states) => Ok(states),
                    None => Err(CheckError::InitCannotEnumerate(
                        "failed to enumerate states from constraints".to_string(),
                    )),
                };
            }
            format!(
                "variable(s) {} have no constraints",
                unconstrained.join(", ")
            )
        } else {
            "Init predicate contains unsupported expressions (only equality, set membership, conjunction, disjunction, and TRUE/FALSE are supported)"
                .to_string()
        };

        // Fallback: enumerate from a bounded type predicate, then filter by the full Init.
        let mut candidates: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        // Common type predicate names
        for name in ["TypeOK", "TypeOk"] {
            if name != pred_name && seen.insert(name) {
                candidates.push(name.to_string());
            }
        }
        // Also consider configured invariants (often includes TypeOK)
        for inv in &self.config.invariants {
            let inv_name = inv.as_str();
            if inv_name != pred_name && seen.insert(inv_name) {
                candidates.push(inv.clone());
            }
        }

        for cand_name in candidates {
            let Some(cand_def) = self.op_defs.get(&cand_name) else {
                continue;
            };
            let Some(branches) = extract_init_constraints(&self.ctx, &cand_def.body, &self.vars)
            else {
                continue;
            };

            let unconstrained = find_unconstrained_vars(&self.vars, &branches);
            if !unconstrained.is_empty() {
                continue;
            }

            let Some(base_states) =
                enumerate_states_from_constraint_branches(Some(&self.ctx), &self.vars, &branches)
            else {
                continue;
            };

            let mut filtered: Vec<State> = Vec::new();
            for state in base_states {
                let saved = self.ctx.save_scope();
                for (name, value) in state.vars() {
                    self.ctx.bind_mut(Arc::clone(name), value.clone());
                }
                let keep = match crate::eval::eval(&self.ctx, &def.body) {
                    Ok(Value::Bool(b)) => b,
                    Ok(_) => {
                        self.ctx.restore_scope(saved);
                        return Err(CheckError::InitNotBoolean);
                    }
                    Err(e) => {
                        self.ctx.restore_scope(saved);
                        return Err(CheckError::EvalError(e));
                    }
                };
                self.ctx.restore_scope(saved);

                if keep {
                    filtered.push(state);
                }
            }

            return Ok(filtered);
        }

        Err(CheckError::InitCannotEnumerate(direct_hint))
    }

    /// Solve a predicate and stream results directly to BulkStateStorage.
    ///
    /// This is a memory-efficient alternative to `solve_predicate_for_states` that avoids
    /// creating intermediate `State` (OrdMap-based) objects. For MCBakery ISpec with 655K states,
    /// this eliminates 655K OrdMap allocations.
    ///
    /// # Returns
    /// - Ok(Some(storage)) - Successfully enumerated states to bulk storage
    /// - Ok(None) - Direct enumeration not possible, caller should fall back to Vec<State> path
    /// - Err(e) - Error during enumeration
    fn solve_predicate_for_states_to_bulk(
        &mut self,
        pred_name: &str,
    ) -> Result<Option<BulkStateStorage>, CheckError> {
        // Resolve operator replacements (e.g., `Init <- MCInit` in config)
        let resolved_name = self.ctx.resolve_op_name(pred_name);

        // Get the operator definition for the predicate
        let def = self
            .op_defs
            .get(resolved_name)
            .ok_or(CheckError::MissingInit)?;
        let pred_body = def.body.clone();

        // Try to extract constraints directly from the Init predicate
        if let Some(branches) = extract_init_constraints(&self.ctx, &pred_body, &self.vars) {
            let unconstrained = find_unconstrained_vars(&self.vars, &branches);
            if unconstrained.is_empty() {
                // Direct enumeration possible - use streaming API
                let vars_len = self.ctx.var_registry().len();
                let mut storage = BulkStateStorage::new(vars_len, 1000);

                // Filter that always accepts (direct constraints already satisfied)
                let count = enumerate_constraints_to_bulk(
                    &mut self.ctx,
                    &self.vars,
                    &branches,
                    &mut storage,
                    |_values, _ctx| Ok(true),
                );

                return match count {
                    Some(_) => Ok(Some(storage)),
                    None => Err(CheckError::InitCannotEnumerate(
                        "failed to stream states from constraints".to_string(),
                    )),
                };
            }
        }

        // Fallback: enumerate from a bounded type predicate, then filter by the full Init
        let mut candidates: Vec<String> = Vec::new();
        let mut seen_cands: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for name in ["TypeOK", "TypeOk"] {
            if name != pred_name && seen_cands.insert(name) {
                candidates.push(name.to_string());
            }
        }
        for inv in &self.config.invariants {
            let inv_name = inv.as_str();
            if inv_name != pred_name && seen_cands.insert(inv_name) {
                candidates.push(inv.clone());
            }
        }

        for cand_name in candidates {
            let Some(cand_def) = self.op_defs.get(&cand_name) else {
                continue;
            };
            let Some(branches) =
                extract_init_constraints(&self.ctx, &cand_def.body.clone(), &self.vars)
            else {
                continue;
            };

            let unconstrained = find_unconstrained_vars(&self.vars, &branches);
            if !unconstrained.is_empty() {
                continue;
            }

            // Optimization: when enumerating from a candidate (like TypeOK), we don't need to
            // re-evaluate the candidate in the filter. Instead, compute the "remainder" of the
            // Init predicate - the parts that are NOT the candidate.
            //
            // Example: If Init = Inv = TypeOK /\ IInv and we're enumerating from TypeOK,
            // we only need to filter by IInv (not the full Inv predicate).
            let filter_expr = extract_conjunction_remainder(&self.ctx, &pred_body, &cand_name)
                .unwrap_or_else(|| pred_body.clone());

            // Check if the remainder is trivially TRUE (no additional filtering needed)
            let filter_is_trivial = matches!(filter_expr.node, tla_core::ast::Expr::Bool(true));

            // Compile the filter expression for fast evaluation using eval_with_values.
            // Use compile_guard_for_filter which compiles quantifiers (ForAll, Exists) and
            // Implies to their optimized variants since eval_with_values uses mutable context.
            let compiled_filter = if filter_is_trivial {
                CompiledGuard::True
            } else {
                compile_guard_for_filter(
                    &self.ctx,
                    &filter_expr,
                    self.ctx.var_registry(),
                    &LocalScope::new(),
                )
            };

            // Stream enumeration with compiled filter
            let vars_len = self.ctx.var_registry().len();
            let mut storage = BulkStateStorage::new(vars_len, 1000);

            let count = enumerate_constraints_to_bulk(
                &mut self.ctx,
                &self.vars,
                &branches,
                &mut storage,
                |values, ctx| {
                    // Use compiled guard with eval_with_values for optimized evaluation.
                    // This uses indexed variable access from the values slice directly,
                    // avoiding string-based lookups in the context.
                    compiled_filter.eval_with_values(ctx, values)
                },
            );

            if count.is_some() {
                return Ok(Some(storage));
            }
        }

        // Streaming not possible - caller should use Vec<State> path
        Ok(None)
    }

    /// Solve next-state relation to find successor states
    fn solve_next_relation(
        &mut self,
        next_name: &str,
        state: &State,
    ) -> Result<Vec<State>, CheckError> {
        let def = self
            .op_defs
            .get(next_name)
            .ok_or(CheckError::MissingNext)?
            .clone();

        let successors = enumerate_successors(&mut self.ctx, &def, state, &self.vars)
            .map_err(CheckError::EvalError)?;

        Ok(successors)
    }

    /// Solve Next relation returning ArrayState instead of State.
    ///
    /// NOTE: Assumes caller has already bound state variables via `bind_state_array`.
    fn solve_next_relation_as_array(
        &mut self,
        current_array: &ArrayState,
    ) -> Result<Vec<ArrayState>, CheckError> {
        let registry = self.ctx.var_registry().clone();
        let state = current_array.to_state(&registry);
        let next_name = self
            .cached_next_name
            .clone()
            .ok_or(CheckError::MissingNext)?;
        let successors = self.solve_next_relation(&next_name, &state)?;
        Ok(successors
            .into_iter()
            .map(|s| ArrayState::from_state(&s, &registry))
            .collect())
    }

    /// Generate successor ArrayStates from a given ArrayState via Next relation.
    fn generate_successors_as_array(
        &mut self,
        current_array: &ArrayState,
    ) -> Result<Vec<ArrayState>, CheckError> {
        let prev_state_env = self.ctx.bind_state_array(current_array.values());

        let successors = self.solve_next_relation_as_array(current_array)?;

        self.ctx.restore_state_env(prev_state_env);

        Ok(successors)
    }

    /// Generate successor DiffSuccessors from a given ArrayState via Next relation.
    ///
    /// Returns None - diff-based enumeration is not currently supported.
    /// Callers fall back to array-based enumeration.
    fn generate_successors_as_diffs(
        &mut self,
        _current_array: &ArrayState,
    ) -> Result<Option<SuccessorResult<Vec<DiffSuccessor>>>, CheckError> {
        Ok(None)
    }

    /// Generate successor ArrayStates filtered by state and action constraints.
    ///
    /// This is the array-based equivalent of `generate_successors_filtered`.
    /// Returns a SuccessorResult that includes whether there were any raw successors
    /// before constraint filtering (used for correct deadlock detection per TLC semantics).
    fn generate_successors_filtered_array(
        &mut self,
        current_array: &ArrayState,
    ) -> Result<SuccessorResult<Vec<ArrayState>>, CheckError> {
        // Coverage collection not supported in array path - use State-based path if enabled
        if self.collect_coverage && !self.coverage_actions.is_empty() {
            let registry = self.ctx.var_registry().clone();
            let state = current_array.to_state(&registry);
            let result = self.generate_successors_filtered("Next", &state)?;
            return Ok(SuccessorResult {
                successors: result
                    .successors
                    .into_iter()
                    .map(|s| ArrayState::from_state(&s, &registry))
                    .collect(),
                had_raw_successors: result.had_raw_successors,
            });
        }

        let successors = self.generate_successors_as_array(current_array)?;
        let had_raw_successors = !successors.is_empty();

        // Fast path: no constraints to check, return successors directly
        if self.config.constraints.is_empty() && self.config.action_constraints.is_empty() {
            return Ok(SuccessorResult {
                successors,
                had_raw_successors,
            });
        }

        // Filter by state constraints and action constraints
        let mut valid = Vec::new();
        let registry = self.ctx.var_registry().clone();

        for succ in successors {
            // Check state constraints using array-based method
            if self.check_state_constraints_array(&succ) {
                // Check action constraints (requires State conversion currently)
                if self.config.action_constraints.is_empty() {
                    valid.push(succ);
                } else {
                    let current_state = current_array.to_state(&registry);
                    let succ_state = succ.to_state(&registry);
                    if self.check_action_constraints(&current_state, &succ_state) {
                        valid.push(succ);
                    }
                }
            }
        }

        Ok(SuccessorResult {
            successors: valid,
            had_raw_successors,
        })
    }

    // ========== State tracking helper methods ==========

    /// Check if a state with the given fingerprint has been seen
    fn is_state_seen(&self, fp: Fingerprint) -> bool {
        self.seen_fps.contains(fp)
    }

    /// Mark a state as seen
    ///
    /// When `store_full_states` is true, stores the ArrayState for trace reconstruction.
    /// When false, only stores the fingerprint for deduplication.
    /// When `trace_file` is enabled, also writes to disk for trace reconstruction.
    fn mark_state_seen(
        &mut self,
        fp: Fingerprint,
        array_state: &ArrayState,
        parent: Option<Fingerprint>,
        depth: usize,
    ) {
        // Always record fingerprints in the scalable fingerprint set, even when
        // storing full states, so checkpoint/resume can restore dedup state
        // without needing to reconstruct all full states in memory.
        self.seen_fps.insert(fp);

        if self.store_full_states {
            self.seen.insert(fp, array_state.clone());
            if let Some(parent_fp) = parent {
                self.parents.insert(fp, parent_fp);
            }
        } else {
            // Write to trace file if enabled
            if let Some(ref mut trace_file) = self.trace_file {
                let loc = if let Some(parent_fp) = parent {
                    // Get parent's location in trace file
                    let parent_loc = self.trace_locs.get(&parent_fp).unwrap_or(0);
                    trace_file.write_state(parent_loc, fp).ok()
                } else {
                    // Initial state - no predecessor
                    trace_file.write_initial(fp).ok()
                };

                if let Some(loc) = loc {
                    self.trace_locs.insert(fp, loc);
                }
            }
        }
        self.depths.insert(fp, depth);
    }

    /// Mark a state as seen when only the fingerprint is available.
    ///
    /// This is intended for no-trace mode, where we don't store full states in memory.
    /// It still supports disk-based trace reconstruction when `trace_file` is enabled.
    fn mark_state_seen_fp_only(
        &mut self,
        fp: Fingerprint,
        parent: Option<Fingerprint>,
        depth: usize,
    ) {
        debug_assert!(!self.store_full_states);

        self.seen_fps.insert(fp);

        // Write to trace file if enabled
        if let Some(ref mut trace_file) = self.trace_file {
            let loc = if let Some(parent_fp) = parent {
                let parent_loc = self.trace_locs.get(&parent_fp).unwrap_or(0);
                trace_file.write_state(parent_loc, fp).ok()
            } else {
                trace_file.write_initial(fp).ok()
            };

            if let Some(loc) = loc {
                self.trace_locs.insert(fp, loc);
            }
        }

        self.depths.insert(fp, depth);
    }

    /// Mark a state as seen, consuming the ArrayState without cloning.
    ///
    /// This is only valid when `store_full_states` is true. Trace-file mode is
    /// mutually exclusive with full-state storage.
    fn mark_state_seen_owned(
        &mut self,
        fp: Fingerprint,
        array_state: ArrayState,
        parent: Option<Fingerprint>,
        depth: usize,
    ) {
        debug_assert!(self.store_full_states);

        // Always record fingerprints in the scalable fingerprint set.
        self.seen_fps.insert(fp);

        self.seen.insert(fp, array_state);
        if let Some(parent_fp) = parent {
            self.parents.insert(fp, parent_fp);
        }
        self.depths.insert(fp, depth);
    }

    /// Get the number of states found (works in both modes)
    fn states_count(&self) -> usize {
        self.seen_fps.len()
    }

    /// Check if the fingerprint storage has encountered any errors (e.g., overflow).
    ///
    /// If errors occurred, returns an error result; otherwise returns None.
    fn check_fingerprint_storage_errors(&self) -> Option<CheckResult> {
        if self.seen_fps.has_errors() {
            let dropped = self.seen_fps.dropped_count();
            Some(CheckResult::Error {
                error: CheckError::FingerprintStorageOverflow { dropped },
                stats: self.stats.clone(),
            })
        } else {
            None
        }
    }

    /// Check fingerprint storage capacity and warn if approaching limits.
    ///
    /// Only emits a warning when the status changes from normal to warning/critical,
    /// or from warning to critical. This avoids spamming the user with repeated warnings.
    fn check_and_warn_capacity(&mut self) {
        let status = self.seen_fps.capacity_status();

        // Only warn if status has changed and is not Normal
        if status == self.last_capacity_status {
            return;
        }

        match status {
            CapacityStatus::Normal => {
                // Status improved back to normal - no warning needed
            }
            CapacityStatus::Warning {
                count,
                capacity,
                usage,
            } => {
                eprintln!(
                    "Warning: Fingerprint storage at {:.1}% capacity ({} / {} states). \
                     Consider increasing --mmap-fingerprints capacity if state space is larger.",
                    usage * 100.0,
                    count,
                    capacity
                );
            }
            CapacityStatus::Critical {
                count,
                capacity,
                usage,
            } => {
                eprintln!(
                    "CRITICAL: Fingerprint storage at {:.1}% capacity ({} / {} states). \
                     Insert failures imminent! Increase --mmap-fingerprints capacity.",
                    usage * 100.0,
                    count,
                    capacity
                );
            }
        }

        self.last_capacity_status = status;
    }

    /// Reconstruct a trace from an initial state to the given state
    ///
    /// When `store_full_states` is true: Uses in-memory parent pointers and state map.
    /// When trace file is enabled: Reads fingerprint path from disk and regenerates states.
    /// Otherwise: Returns an empty trace.
    fn reconstruct_trace(&mut self, end_fp: Fingerprint) -> Trace {
        // Case 1: Full states stored in memory (as ArrayState)
        if self.store_full_states {
            let fps = self.fingerprint_path_from_parents(end_fp);
            let registry = self.ctx.var_registry().clone();

            // Fast path: all states are present in memory (convert ArrayState to State).
            let states: Vec<State> = fps
                .iter()
                .filter_map(|fp| self.seen.get(fp).map(|arr| arr.to_state(&registry)))
                .collect();
            if states.len() == fps.len() {
                return Trace::from_states(states);
            }

            // Fallback: checkpoint/resume may restore fingerprints+parents without full states.
            // In that case, reconstruct by replaying from Init/Next using the fingerprint path.
            return self.reconstruct_trace_from_fingerprint_path(&fps);
        }

        // Case 2: Trace file enabled - reconstruct from fingerprints
        if self.trace_file.is_some() {
            return self.reconstruct_trace_from_file(end_fp);
        }

        // Case 3: No-trace mode - return empty trace
        Trace::new()
    }

    /// Reconstruct a trace from a trace file by replaying from initial state.
    ///
    /// This is slower than in-memory trace storage but enables trace reconstruction
    /// for large state spaces that exceed available RAM.
    fn reconstruct_trace_from_file(&mut self, end_fp: Fingerprint) -> Trace {
        // Get the location of the end state in the trace file
        let Some(end_loc) = self.trace_locs.get(&end_fp) else {
            // Fingerprint not found in trace locations - can't reconstruct
            return Trace::new();
        };

        // Get the fingerprint path from the trace file
        let fingerprint_path = {
            let Some(ref mut trace_file) = self.trace_file else {
                return Trace::new();
            };
            match trace_file.get_fingerprint_path(end_loc) {
                Ok(path) => path,
                Err(_) => return Trace::new(),
            }
        };

        self.reconstruct_trace_from_fingerprint_path(&fingerprint_path)
    }

    /// Compute the fingerprint path from an initial state to `end_fp` using parent pointers.
    fn fingerprint_path_from_parents(&self, end_fp: Fingerprint) -> Vec<Fingerprint> {
        let mut fps = Vec::new();
        let mut current = end_fp;

        fps.push(current);
        while let Some(&parent) = self.parents.get(&current) {
            current = parent;
            fps.push(current);
        }

        fps.reverse();
        fps
    }

    /// Reconstruct a trace by replaying from the initial state, matching a known fingerprint path.
    ///
    /// This is used for trace-file mode and as a fallback for checkpoint/resume when full states
    /// are not available in memory.
    fn reconstruct_trace_from_fingerprint_path(
        &mut self,
        fingerprint_path: &[Fingerprint],
    ) -> Trace {
        if fingerprint_path.is_empty() {
            return Trace::new();
        }

        // We need Init and Next names to regenerate states
        let (init_name, next_name) = match (&self.cached_init_name, &self.cached_next_name) {
            (Some(init), Some(next)) => (init.clone(), next.clone()),
            _ => return Trace::new(), // Names not cached - can't replay
        };

        // Replay from initial state, matching by fingerprint
        let mut states = Vec::new();

        // Generate initial states and find the one matching the first fingerprint
        let target_init_fp = fingerprint_path[0];
        let initial_states = match self.generate_initial_states(&init_name) {
            Ok(s) => s,
            Err(_) => return Trace::new(),
        };

        let Some(mut current_state) = initial_states
            .into_iter()
            .find(|s| self.state_fingerprint(s) == target_init_fp)
        else {
            return Trace::new();
        };

        states.push(current_state.clone());

        // For each subsequent fingerprint, generate successors and find the match
        for &target_fp in &fingerprint_path[1..] {
            let successors = match self.solve_next_relation(&next_name, &current_state) {
                Ok(s) => s,
                Err(_) => return Trace::from_states(states), // Partial trace
            };

            let Some(next_state) = successors
                .into_iter()
                .find(|s| self.state_fingerprint(s) == target_fp)
            else {
                return Trace::from_states(states); // Partial trace
            };

            states.push(next_state.clone());
            current_state = next_state;
        }

        Trace::from_states(states)
    }

    /// Convert a fairness constraint to a LiveExpr
    ///
    /// WF_vars(Action) becomes []<>(~ENABLED(Action) \/ Action)
    /// SF_vars(Action) becomes <>[]~ENABLED(Action) \/ []<>Action
    /// TemporalRef directly converts the referenced operator's body to LiveExpr
    fn fairness_to_live_expr(
        &self,
        constraint: &FairnessConstraint,
        converter: &AstToLive,
    ) -> Result<LiveExpr, String> {
        // Handle TemporalRef separately - it directly converts the operator body
        if let FairnessConstraint::TemporalRef { op_name } = constraint {
            let def = self.op_defs.get(op_name).ok_or_else(|| {
                format!("Operator '{}' not found for temporal reference", op_name)
            })?;
            return converter
                .convert(&self.ctx, &def.body)
                .map_err(|e| format!("Failed to convert temporal formula '{}': {}", op_name, e));
        }

        // Handle QuantifiedTemporal - inline quantified fairness from spec body
        // This handles cases like `\A c \in Clients: WF_vars(Return(c, alloc[c]))`
        if let FairnessConstraint::QuantifiedTemporal { node } = constraint {
            // Lower the syntax node directly to AST
            let expr = tla_core::lower_single_expr(tla_core::FileId(0), node)
                .ok_or_else(|| "Failed to lower inline quantified temporal formula".to_string())?;
            let spanned = Spanned::dummy(expr);
            return converter
                .convert(&self.ctx, &spanned)
                .map_err(|e| format!("Failed to convert quantified temporal formula: {}", e));
        }

        let (is_weak, action_name, action_node, vars_name) = match constraint {
            FairnessConstraint::Weak {
                action,
                action_node,
                vars,
            } => (true, action, action_node, vars),
            FairnessConstraint::Strong {
                action,
                action_node,
                vars,
            } => (false, action, action_node, vars),
            FairnessConstraint::TemporalRef { .. }
            | FairnessConstraint::QuantifiedTemporal { .. } => {
                unreachable!("Handled above")
            }
        };

        // Get the action expression body either from op_defs or by lowering the inline expression
        let action_body: Spanned<Expr> = if let Some(def) = self.op_defs.get(action_name) {
            // Simple case: action is a named operator
            def.body.clone()
        } else if let Some(node) = action_node {
            // Inline expression case: lower the syntax node directly
            let expr = tla_core::lower_single_expr(tla_core::FileId(0), node).ok_or_else(|| {
                format!(
                    "Failed to lower inline action expression for fairness constraint: {}",
                    action_name
                )
            })?;
            // Wrap in Spanned with dummy span since we don't have source location
            Spanned::dummy(expr)
        } else {
            return Err(format!(
                "Action '{}' not found for fairness constraint",
                action_name
            ));
        };

        // Get the subscript expression (vars) - used for stuttering check
        // For WF_vars(A), we need to check if vars' != vars, not global state change
        let subscript_expr: Option<Arc<Spanned<Expr>>> =
            if let Some(def) = self.op_defs.get(vars_name) {
                Some(Arc::new(def.body.clone()))
            } else if vars_name.starts_with("<<") {
                // vars_name is a tuple expression like "<<coordinator, participant>>"
                // Parse it as a tuple of identifiers
                let inner = vars_name
                    .trim_start_matches("<<")
                    .trim_end_matches(">>");
                let var_names: Vec<_> = inner
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();
                let tuple_elems: Vec<_> = var_names
                    .iter()
                    .map(|vn| Spanned::dummy(Expr::Ident((*vn).into())))
                    .collect();
                Some(Arc::new(Spanned::dummy(Expr::Tuple(tuple_elems))))
            } else {
                // vars_name is a simple identifier like "vars"
                // Create an identifier expression for it
                Some(Arc::new(Spanned::dummy(Expr::Ident(vars_name.clone()))))
            };

        // Convert action to LiveExpr (as an action predicate)
        let action_live = match converter.convert(&self.ctx, &action_body) {
            Ok(live) => live,
            Err(e) => {
                return Err(format!(
                    "Failed to convert action '{}' for fairness: {}",
                    action_name, e
                ));
            }
        };

        // Create ENABLED(<<A>>_vars) predicate (i.e., only enabled if the action can occur
        // with a non-stuttering transition).
        // Use converter.alloc_tag() for unique tags to avoid tag collisions with other LiveExprs.
        let enabled = LiveExpr::enabled_subscripted(
            Arc::new(action_body),
            subscript_expr.clone(),
            converter.alloc_tag(),
        );

        // Create subscripted action <<A>>_vars = A /\ (vars' ≠ vars)
        // StateChanged checks if the subscript expression changed (e' ≠ e)
        let state_changed = LiveExpr::state_changed(subscript_expr, converter.alloc_tag());
        let subscripted_action = LiveExpr::and(vec![action_live, state_changed]);

        if is_weak {
            // WF_vars(Action): []<>(~ENABLED(<<A>>_vars) \/ <<A>>_vars)
            // Always eventually: either action is not enabled, or action happens (with state change)
            let not_enabled = LiveExpr::not(enabled);
            let disj = LiveExpr::or(vec![not_enabled, subscripted_action]);
            Ok(LiveExpr::always(LiveExpr::eventually(disj)))
        } else {
            // SF_vars(Action): <>[]~ENABLED(<<A>>_vars) \/ []<><<A>>_vars
            // Eventually always disabled, or infinitely often happens (with state change)
            let not_enabled = LiveExpr::not(enabled);
            let eventually_always_disabled = LiveExpr::eventually(LiveExpr::always(not_enabled));
            let infinitely_often = LiveExpr::always(LiveExpr::eventually(subscripted_action));
            Ok(LiveExpr::or(vec![
                eventually_always_disabled,
                infinitely_often,
            ]))
        }
    }

    /// Check a single liveness property
    ///
    /// Returns `Some(CheckResult)` if the property is violated, `None` if satisfied.
    fn check_liveness_property(
        &mut self,
        prop_name: &str,
        init_states: &[State],
        state_successors: &Arc<FxHashMap<Fingerprint, Vec<State>>>,
        state_fp_to_canon_fp: &Arc<FxHashMap<Fingerprint, Fingerprint>>,
        succ_witnesses: Option<&Arc<SuccessorWitnessMap>>,
    ) -> Option<CheckResult> {
        let func_start = std::time::Instant::now();
        // Get the property operator definition
        let def = match self.op_defs.get(prop_name) {
            Some(d) => d.clone(),
            None => {
                return Some(CheckResult::Error {
                    error: CheckError::MissingProperty(prop_name.to_string()),
                    stats: self.stats.clone(),
                });
            }
        };

        // Separate safety and liveness parts of the property BEFORE converting to LiveExpr.
        //
        // A property like `Init /\ [][Next]_vars /\ Liveness` has:
        // - Safety parts: Init (checked on initial states), [][Next]_vars (checked on transitions)
        // - Liveness parts: e.g., []<>P, WF, SF
        //
        // We check safety parts inline and only pass liveness parts to the liveness checker.
        // This avoids the "unsupported temporal subformula containing actions" error that
        // occurs when negating [][Next]_vars (which becomes <>~[Next]_vars).
        let (safety_parts, liveness_expr) =
            self.separate_safety_liveness_parts(prop_name, &def.body, state_successors)?;

        // Check safety parts: init terms on initial states, always terms on transitions
        if let Some(result) = self.check_property_safety_parts(
            prop_name,
            &safety_parts,
            init_states,
            state_successors,
            state_fp_to_canon_fp,
            succ_witnesses,
        ) {
            return Some(result);
        }

        // If no liveness parts, property is satisfied
        let liveness_expr = liveness_expr?;

        // Convert the liveness-only part to LiveExpr
        let converter = AstToLive::new();
        let prop_live = match converter.convert(&self.ctx, &liveness_expr) {
            Ok(live) => live,
            Err(e) => {
                return Some(CheckResult::Error {
                    error: CheckError::LivenessError(format!(
                        "Failed to convert property '{}': {}",
                        prop_name, e
                    )),
                    stats: self.stats.clone(),
                });
            }
        };

        // For liveness checking, we check: fairness /\ ~property
        // If this is satisfiable (has an accepting cycle), the property is violated.
        let negated_prop = LiveExpr::not(prop_live).push_negation();

        // Build formula: fairness /\ ~property
        // If there are fairness constraints, conjoin them with the negated property
        let formula = if self.fairness.is_empty() {
            negated_prop
        } else {
            // Convert fairness constraints to LiveExpr
            let mut fairness_exprs: Vec<LiveExpr> = Vec::new();
            for constraint in &self.fairness {
                match self.fairness_to_live_expr(constraint, &converter) {
                    Ok(expr) => fairness_exprs.push(expr),
                    Err(e) => {
                        return Some(CheckResult::Error {
                            error: CheckError::LivenessError(format!(
                                "Failed to process fairness for property '{}': {}",
                                prop_name, e
                            )),
                            stats: self.stats.clone(),
                        });
                    }
                }
            }
            // Conjoin all fairness with negated property: fairness1 /\ fairness2 /\ ... /\ ~property
            fairness_exprs.push(negated_prop);
            LiveExpr::and(fairness_exprs).push_negation()
        };

        // Create liveness checkers from the DNF decomposition
        if std::env::var("TLA2_DEBUG_LIVENESS_FORMULA").is_ok() {
            eprintln!("[DEBUG FORMULA] Final formula: {}", formula);
        }
        let checkers = match LivenessChecker::from_formula(&formula, self.ctx.clone()) {
            Ok(cs) => cs,
            Err(e) => {
                return Some(CheckResult::Error {
                    error: CheckError::LivenessError(format!(
                        "Failed to create liveness checker for '{}': {}",
                        prop_name, e
                    )),
                    stats: self.stats.clone(),
                });
            }
        };

        // Run each checker (DNF clause) - a violation in any clause is a property violation
        if std::env::var("TLA2_DEBUG_LIVENESS_FORMULA").is_ok() {
            eprintln!("[DEBUG FORMULA] Created {} liveness checkers (DNF clauses)", checkers.len());
            for (i, checker) in checkers.iter().enumerate() {
                eprintln!("[DEBUG FORMULA] Checker {}: tableau={}, promises={}", i, checker.tableau().formula(), checker.promises().len());
                let c = checker.constraints();
                eprintln!("[DEBUG FORMULA]   ae_state: {:?}", c.ae_state.iter().map(|e| e.to_string()).collect::<Vec<_>>());
                eprintln!("[DEBUG FORMULA]   ae_action: {:?}", c.ae_action.iter().map(|e| e.to_string()).collect::<Vec<_>>());
                eprintln!("[DEBUG FORMULA]   ea_state: {:?}", c.ea_state.iter().map(|e| e.to_string()).collect::<Vec<_>>());
                eprintln!("[DEBUG FORMULA]   ea_action: {:?}", c.ea_action.iter().map(|e| e.to_string()).collect::<Vec<_>>());
                // Print tableau structure
                let tableau = checker.tableau();
                eprintln!("[DEBUG FORMULA]   tableau_nodes: {} (init: {})", tableau.len(), tableau.init_count());
                for node in tableau.nodes() {
                    eprintln!("[DEBUG FORMULA]     node {}: state_preds={:?}",
                        node.index(),
                        node.state_preds().iter().map(|p| p.to_string()).collect::<Vec<_>>()
                    );
                }
            }
        }
        for (checker_idx, mut checker) in checkers.into_iter().enumerate() {
            if std::env::var("TLA2_DEBUG_LIVENESS_FORMULA").is_ok() {
                eprintln!("[DEBUG] Starting checker {} for liveness", checker_idx);
            }
            checker.set_successor_maps(
                Arc::clone(state_fp_to_canon_fp),
                Arc::clone(state_successors),
                succ_witnesses.map(Arc::clone),
            );

            // Build the behavior graph
            // Stuttering edges:
            // - With SPECIFICATION, we mirror TLC: we do NOT add implicit stuttering edges.
            //   Terminal states represent finite behaviors, which don't violate liveness.
            // - With only INIT/NEXT (no SPECIFICATION), TLC performs "implied-temporal checking"
            //   by treating the behavior as `Init /\ [][Next]_vars`, i.e., stuttering is always
            //   permitted. In that mode, deadlocks extend to infinite stuttering and can violate
            //   temporal properties (see TLC warning in MCRealTimeHourClock).
            let implied_temporal = self.config.specification.is_none();
            let mut get_successors = |state: &State| {
                let fp = self.state_fingerprint(state);
                let mut succs = state_successors.get(&fp).cloned().unwrap_or_default();
                if implied_temporal {
                    succs.push(state.clone());
                }
                Ok(succs)
            };

            if let Err(e) = checker.explore_bfs(init_states, &mut get_successors) {
                return Some(CheckResult::Error {
                    error: CheckError::LivenessError(format!(
                        "Error during liveness exploration for '{}': {}",
                        prop_name, e
                    )),
                    stats: self.stats.clone(),
                });
            }

            // Print timing statistics if enabled via LIVENESS_PROFILE env var
            if std::env::var("LIVENESS_PROFILE").is_ok() {
                let stats = checker.stats();
                eprintln!("=== Liveness profiling ({}) ===", prop_name);
                eprintln!(
                    "  init_state_time:     {:.3}s",
                    stats.init_state_time_us as f64 / 1_000_000.0
                );
                eprintln!(
                    "  state_clone_time:    {:.3}s",
                    stats.state_clone_time_us as f64 / 1_000_000.0
                );
                eprintln!(
                    "  get_successors_time: {:.3}s",
                    stats.get_successors_time_us as f64 / 1_000_000.0
                );
                eprintln!(
                    "  add_successors_time: {:.3}s",
                    stats.add_successors_time_us as f64 / 1_000_000.0
                );
                eprintln!("  consistency_checks:  {}", stats.consistency_checks);
                eprintln!("  graph_nodes:         {}", stats.graph_nodes);
                eprintln!("  graph_edges:         {}", stats.graph_edges);
                eprintln!("===================================");
            }

            // Check for liveness violations
            let check_start = std::time::Instant::now();
            let check_result = checker.check_liveness();
            if std::env::var("LIVENESS_PROFILE").is_ok() {
                eprintln!(
                    "  check_liveness time: {:.3}s",
                    check_start.elapsed().as_secs_f64()
                );
            }
            match check_result {
                LivenessResult::Satisfied => {
                    // This DNF clause is satisfied, continue to next
                }
                LivenessResult::Violated {
                    prefix,
                    cycle,
                    property_desc: _,
                } => {
                    // Convert (State, tableau_idx) pairs to just State for the trace
                    let prefix_trace =
                        Trace::from_states(prefix.into_iter().map(|(s, _)| s).collect());
                    let cycle_trace =
                        Trace::from_states(cycle.into_iter().map(|(s, _)| s).collect());

                    return Some(CheckResult::LivenessViolation {
                        property: prop_name.to_string(),
                        prefix: prefix_trace,
                        cycle: cycle_trace,
                        stats: self.stats.clone(),
                    });
                }
                LivenessResult::Incomplete { reason } => {
                    return Some(CheckResult::Error {
                        error: CheckError::LivenessError(format!(
                            "Liveness check incomplete for '{}': {}",
                            prop_name, reason
                        )),
                        stats: self.stats.clone(),
                    });
                }
            }
        }

        // All DNF clauses satisfied - property holds
        if std::env::var("LIVENESS_PROFILE").is_ok() {
            eprintln!(
                "Total check_liveness_property time: {:.3}s",
                func_start.elapsed().as_secs_f64()
            );
        }
        None
    }

    /// Check an invariant that contains ENABLED via the liveness checker.
    ///
    /// Invariants containing ENABLED cannot be evaluated by the normal eval() path
    /// because ENABLED requires knowledge of successor states. Instead, we check them
    /// as `[]Invariant` (Always holds) using the liveness checking infrastructure.
    ///
    /// Returns `Some(CheckResult)` if the invariant is violated, `None` if satisfied.
    fn check_enabled_invariant(
        &mut self,
        inv_name: &str,
        init_states: &[State],
        state_successors: &Arc<FxHashMap<Fingerprint, Vec<State>>>,
        state_fp_to_canon_fp: &Arc<FxHashMap<Fingerprint, Fingerprint>>,
        succ_witnesses: Option<&Arc<SuccessorWitnessMap>>,
    ) -> Option<CheckResult> {
        // Get the invariant operator definition
        let def = match self.op_defs.get(inv_name) {
            Some(d) => d.clone(),
            None => {
                return Some(CheckResult::Error {
                    error: CheckError::MissingInvariant(inv_name.to_string()),
                    stats: self.stats.clone(),
                });
            }
        };

        // Wrap the invariant in [] (Always) to create the temporal formula []Invariant
        let always_expr = Spanned::new(Expr::Always(Box::new(def.body.clone())), def.body.span);

        // Convert to LiveExpr
        let converter = AstToLive::new();
        let inv_live = match converter.convert(&self.ctx, &always_expr) {
            Ok(live) => live,
            Err(e) => {
                return Some(CheckResult::Error {
                    error: CheckError::LivenessError(format!(
                        "Failed to convert ENABLED-containing invariant '{}': {}",
                        inv_name, e
                    )),
                    stats: self.stats.clone(),
                });
            }
        };

        // For liveness checking, we check: ~[]Invariant (negated formula)
        // If this is satisfiable (has an accepting cycle), the invariant is violated.
        let negated_inv = LiveExpr::not(inv_live).push_negation();

        // Create liveness checkers from the DNF decomposition
        let checkers = match LivenessChecker::from_formula(&negated_inv, self.ctx.clone()) {
            Ok(cs) => cs,
            Err(e) => {
                return Some(CheckResult::Error {
                    error: CheckError::LivenessError(format!(
                        "Failed to create liveness checker for ENABLED invariant '{}': {}",
                        inv_name, e
                    )),
                    stats: self.stats.clone(),
                });
            }
        };

        // Run each checker (DNF clause) - a violation in any clause is an invariant violation
        for mut checker in checkers {
            checker.set_successor_maps(
                Arc::clone(state_fp_to_canon_fp),
                Arc::clone(state_successors),
                succ_witnesses.map(Arc::clone),
            );

            // Build the behavior graph using explore_bfs
            // Include stuttering edges for completeness (TLA+ allows infinite stuttering)
            let mut get_successors = |state: &State| {
                let fp = self.state_fingerprint(state);
                let mut succs = state_successors.get(&fp).cloned().unwrap_or_default();
                // Add stuttering edge (self-loop)
                succs.push(state.clone());
                Ok(succs)
            };
            if let Err(e) = checker.explore_bfs(init_states, &mut get_successors) {
                return Some(CheckResult::Error {
                    error: CheckError::LivenessError(format!(
                        "Error during ENABLED invariant exploration for '{}': {}",
                        inv_name, e
                    )),
                    stats: self.stats.clone(),
                });
            }

            // Check for liveness violations
            let check_result = checker.check_liveness();
            match check_result {
                LivenessResult::Satisfied => {
                    // This DNF clause is satisfied, continue to next
                }
                LivenessResult::Violated {
                    prefix,
                    cycle: _,
                    property_desc: _,
                } => {
                    // For invariants, we report an invariant violation
                    // The prefix shows the path to the violating state
                    // Use the last state in the prefix as the counterexample state
                    let trace = Trace::from_states(prefix.into_iter().map(|(s, _)| s).collect());

                    return Some(CheckResult::InvariantViolation {
                        invariant: inv_name.to_string(),
                        trace,
                        stats: self.stats.clone(),
                    });
                }
                LivenessResult::Incomplete { reason } => {
                    return Some(CheckResult::Error {
                        error: CheckError::LivenessError(format!(
                            "ENABLED invariant check incomplete for '{}': {}",
                            inv_name, reason
                        )),
                        stats: self.stats.clone(),
                    });
                }
            }
        }

        // All DNF clauses satisfied - invariant holds
        None
    }

    /// Separate a property into safety parts (checkable on transitions) and liveness parts.
    ///
    /// Safety parts include:
    /// - Init predicates (state-level, no temporal operators)
    /// - Always-action formulas: `[]Action` where Action is action-level (may contain primes)
    ///   but has no nested temporal operators
    ///
    /// Liveness parts include:
    /// - `[]<>P` (infinitely often)
    /// - `<>[]P` (eventually always)
    /// - `WF_v(A)`, `SF_v(A)` (fairness)
    /// - `P ~> Q` (leads-to)
    /// - Any formula with nested temporal operators
    ///
    /// Returns (safety_parts, liveness_expr) where:
    /// - safety_parts contains init_terms and always_terms for safety checking
    /// - liveness_expr is Some(conjunction of liveness terms) or None if no liveness
    fn separate_safety_liveness_parts(
        &self,
        _prop_name: &str,
        body: &Spanned<Expr>,
        _state_successors: &FxHashMap<Fingerprint, Vec<State>>,
    ) -> Option<(PropertySafetyParts, Option<Spanned<Expr>>)> {
        // Local helper: check if expression contains temporal operators (simplified)
        fn contains_temporal_local(expr: &Expr) -> bool {
            match expr {
                Expr::Always(_)
                | Expr::Eventually(_)
                | Expr::LeadsTo(_, _)
                | Expr::WeakFair(_, _)
                | Expr::StrongFair(_, _) => true,
                Expr::And(l, r) | Expr::Or(l, r) | Expr::Implies(l, r) | Expr::Equiv(l, r) => {
                    contains_temporal_local(&l.node) || contains_temporal_local(&r.node)
                }
                Expr::Not(e) => contains_temporal_local(&e.node),
                Expr::Apply(op, args) => {
                    if let Expr::Ident(name) = &op.node {
                        if name.starts_with("WF_") || name.starts_with("SF_") {
                            return true;
                        }
                    }
                    contains_temporal_local(&op.node)
                        || args.iter().any(|a| contains_temporal_local(&a.node))
                }
                Expr::Forall(_, body) | Expr::Exists(_, body) => {
                    contains_temporal_local(&body.node)
                }
                _ => false,
            }
        }

        // Local helper: check if expression contains ENABLED operator.
        // ENABLED cannot be evaluated by eval() - it must go through the liveness
        // checker which has special handling via eval_live_expr.
        fn contains_enabled_local(expr: &Expr) -> bool {
            match expr {
                Expr::Enabled(_) => true,
                Expr::And(l, r) | Expr::Or(l, r) | Expr::Implies(l, r) | Expr::Equiv(l, r) => {
                    contains_enabled_local(&l.node) || contains_enabled_local(&r.node)
                }
                Expr::Not(e) => contains_enabled_local(&e.node),
                Expr::Apply(op, args) => {
                    contains_enabled_local(&op.node)
                        || args.iter().any(|a| contains_enabled_local(&a.node))
                }
                Expr::Forall(_, body) | Expr::Exists(_, body) => contains_enabled_local(&body.node),
                Expr::If(cond, then_e, else_e) => {
                    contains_enabled_local(&cond.node)
                        || contains_enabled_local(&then_e.node)
                        || contains_enabled_local(&else_e.node)
                }
                Expr::Let(defs, body) => {
                    defs.iter().any(|d| contains_enabled_local(&d.body.node))
                        || contains_enabled_local(&body.node)
                }
                Expr::Always(inner) | Expr::Eventually(inner) => {
                    contains_enabled_local(&inner.node)
                }
                _ => false,
            }
        }

        // Helper to check if expression has nested temporal operators
        fn has_nested_temporal(expr: &Expr) -> bool {
            match expr {
                // Temporal operators - check if their body also has temporal
                Expr::Always(inner) | Expr::Eventually(inner) => {
                    contains_temporal_local(&inner.node)
                }
                Expr::LeadsTo(_, _) | Expr::WeakFair(_, _) | Expr::StrongFair(_, _) => {
                    // These are inherently "nested temporal" patterns
                    true
                }
                // Non-temporal cases don't have nested temporal
                Expr::Bool(_)
                | Expr::Int(_)
                | Expr::String(_)
                | Expr::Ident(_)
                | Expr::OpRef(_)
                | Expr::Prime(_)
                | Expr::Unchanged(_)
                | Expr::Enabled(_) => false,
                // Recurse for compound expressions
                Expr::And(a, b) | Expr::Or(a, b) | Expr::Implies(a, b) | Expr::Equiv(a, b) => {
                    has_nested_temporal(&a.node) || has_nested_temporal(&b.node)
                }
                Expr::Not(e) => has_nested_temporal(&e.node),
                _ => contains_temporal_local(expr),
            }
        }

        // Flatten conjunction
        fn flatten_and_terms(expr: &Spanned<Expr>, out: &mut Vec<Spanned<Expr>>) {
            match &expr.node {
                Expr::And(left, right) => {
                    flatten_and_terms(left, out);
                    flatten_and_terms(right, out);
                }
                _ => out.push(expr.clone()),
            }
        }

        let mut terms = Vec::new();
        flatten_and_terms(body, &mut terms);

        let mut init_terms: Vec<Spanned<Expr>> = Vec::new();
        let mut always_terms: Vec<Spanned<Expr>> = Vec::new();
        let mut liveness_terms: Vec<Spanned<Expr>> = Vec::new();

        for term in terms {
            match &term.node {
                // Always formula: []inner
                Expr::Always(inner) => {
                    if has_nested_temporal(&term.node) {
                        // Has nested temporal (e.g., []<>P) - goes to liveness
                        liveness_terms.push(term.clone());
                    } else if contains_enabled_local(&inner.node) {
                        // Contains ENABLED - must go to liveness because eval() can't handle it
                        liveness_terms.push(term.clone());
                    } else {
                        // Pure safety: []Action where Action has no temporal and no ENABLED
                        always_terms.push((**inner).clone());
                    }
                }
                // Eventually, leads-to, fairness are liveness
                Expr::Eventually(_)
                | Expr::LeadsTo(_, _)
                | Expr::WeakFair(_, _)
                | Expr::StrongFair(_, _) => {
                    liveness_terms.push(term.clone());
                }
                // Check for WF_xxx/SF_xxx parsed as Apply
                Expr::Apply(op, _) => {
                    if let Expr::Ident(name) = &op.node {
                        if name.starts_with("WF_") || name.starts_with("SF_") {
                            liveness_terms.push(term.clone());
                            continue;
                        }
                    }
                    // Other Apply - check if it contains temporal
                    if Self::contains_temporal_helper(&term.node) {
                        liveness_terms.push(term.clone());
                    } else {
                        init_terms.push(term.clone());
                    }
                }
                // Identifier - might be a reference to a temporal operator or conjunction
                Expr::Ident(name) => {
                    // Look up the operator to check its body
                    if let Some(op_def) = self.op_defs.get(name) {
                        // If it's a conjunction, recursively flatten it
                        if let Expr::And(_, _) = &op_def.body.node {
                            // Recursively process the expanded definition
                            let mut expanded = Vec::new();
                            flatten_and_terms(&op_def.body, &mut expanded);
                            for subterm in expanded {
                                match &subterm.node {
                                    Expr::Always(inner) => {
                                        if has_nested_temporal(&subterm.node) {
                                            liveness_terms.push(subterm.clone());
                                        } else if contains_enabled_local(&inner.node) {
                                            // Contains ENABLED - must go to liveness
                                            liveness_terms.push(subterm.clone());
                                        } else {
                                            always_terms.push((**inner).clone());
                                        }
                                    }
                                    Expr::Eventually(_)
                                    | Expr::LeadsTo(_, _)
                                    | Expr::WeakFair(_, _)
                                    | Expr::StrongFair(_, _) => {
                                        liveness_terms.push(subterm.clone());
                                    }
                                    _ => {
                                        if contains_temporal_local(&subterm.node)
                                            || contains_enabled_local(&subterm.node)
                                        {
                                            liveness_terms.push(subterm.clone());
                                        } else {
                                            init_terms.push(subterm.clone());
                                        }
                                    }
                                }
                            }
                        } else if Self::contains_temporal_helper(&op_def.body.node)
                            || contains_enabled_local(&op_def.body.node)
                        {
                            // References a temporal operator or contains ENABLED - check if it's liveness
                            if has_nested_temporal(&op_def.body.node)
                                || contains_enabled_local(&op_def.body.node)
                            {
                                liveness_terms.push(term.clone());
                            } else if let Expr::Always(inner) = &op_def.body.node {
                                // It's a pure []Action - extract as safety
                                always_terms.push((**inner).clone());
                            } else {
                                liveness_terms.push(term.clone());
                            }
                        } else {
                            init_terms.push(term.clone());
                        }
                    } else {
                        init_terms.push(term.clone());
                    }
                }
                // ModuleRef - references an operator from an INSTANCE.
                //
                // Prefer splitting spec-shaped instance references into safety/liveness parts:
                // this avoids sending `[][Action]_vars` through the liveness tableau, which
                // is both expensive and currently handled via a (slow) <>Action heuristic.
                //
                // Fallback: route to liveness when we can't safely split.
                Expr::ModuleRef(instance_target, op_name, args) => {
                    // Only attempt the split for zero-argument module refs into a *named* instance.
                    if args.is_empty() {
                        if let tla_core::ast::ModuleTarget::Named(instance_name) = instance_target {
                            if let Some(info) = self.ctx.get_instance(instance_name) {
                                if let Some(op_def) =
                                    self.ctx.get_instance_op(&info.module_name, op_name)
                                {
                                    if op_def.params.is_empty() {
                                        // Inline the referenced operator body with the INSTANCE substitutions
                                        // so that extracted safety terms are evaluable in the outer module.
                                        let substituted_body = crate::eval::apply_substitutions(
                                            &op_def.body,
                                            &info.substitutions,
                                        );

                                        fn flatten_and_terms(
                                            expr: &Spanned<Expr>,
                                            out: &mut Vec<Spanned<Expr>>,
                                        ) {
                                            match &expr.node {
                                                Expr::And(left, right) => {
                                                    flatten_and_terms(left, out);
                                                    flatten_and_terms(right, out);
                                                }
                                                _ => out.push(expr.clone()),
                                            }
                                        }

                                        let mut parts = Vec::new();
                                        flatten_and_terms(&substituted_body, &mut parts);

                                        // Track if we successfully extracted any safety terms; if not,
                                        // fall back to treating the module ref as liveness.
                                        let mut extracted_any = false;

                                        for part in parts {
                                            match &part.node {
                                                // Handle bare identifiers - check if they reference temporal formulas.
                                                // `Init` -> state-level -> init_terms
                                                // `Liveness` (containing WF/SF) -> temporal -> liveness_terms
                                                Expr::Ident(name) => {
                                                    // Look up the operator in the instance module to check if it contains temporal operators
                                                    let is_temporal = self
                                                        .ctx
                                                        .get_instance_op(&info.module_name, name)
                                                        .is_some_and(|op_def| {
                                                            contains_temporal_local(
                                                                &op_def.body.node,
                                                            )
                                                        });

                                                    let module_ref = Spanned {
                                                        node: Expr::ModuleRef(
                                                            instance_target.clone(),
                                                            name.clone(),
                                                            Vec::new(),
                                                        ),
                                                        span: part.span,
                                                    };

                                                    if is_temporal {
                                                        // Route temporal formulas (e.g., Liveness with WF/SF) to liveness checker
                                                        liveness_terms.push(module_ref);
                                                    } else {
                                                        // Non-temporal identifiers (e.g., Init) are init terms
                                                        init_terms.push(module_ref);
                                                    }
                                                    extracted_any = true;
                                                }
                                                // Treat []Action as an always-action safety term. For specs
                                                // written as `[][Next]_vars`, the lowered form is:
                                                //   [](Next \/ UNCHANGED vars)
                                                // We preserve the full `Next \/ UNCHANGED vars` semantics by
                                                // qualifying `Next`/`vars` into the instance scope.
                                                Expr::Always(inner) => {
                                                    let extracted_action: Option<Spanned<Expr>> =
                                                        match &inner.node {
                                                            Expr::Or(left, right) => {
                                                                let (
                                                                    action,
                                                                    unchanged,
                                                                    action_on_left,
                                                                ) = match (&left.node, &right.node)
                                                                {
                                                                    (Expr::Unchanged(_), _) => (
                                                                        right.as_ref(),
                                                                        left.as_ref(),
                                                                        false,
                                                                    ),
                                                                    (_, Expr::Unchanged(_)) => (
                                                                        left.as_ref(),
                                                                        right.as_ref(),
                                                                        true,
                                                                    ),
                                                                    _ => (
                                                                        inner.as_ref(),
                                                                        inner.as_ref(),
                                                                        true,
                                                                    ),
                                                                };

                                                                if let Expr::Unchanged(unch_inner) =
                                                                    &unchanged.node
                                                                {
                                                                    let qualified_action = match &action.node {
		                                                                    Expr::Ident(action_name) => Spanned {
		                                                                        node: Expr::ModuleRef(
		                                                                            instance_target.clone(),
		                                                                            action_name.clone(),
		                                                                            Vec::new(),
		                                                                        ),
		                                                                        span: action.span,
		                                                                    },
		                                                                    _ => action.clone(),
		                                                                };

                                                                    let qualified_unch_inner = match &unch_inner.node {
		                                                                    Expr::Ident(v) => Spanned {
		                                                                        node: Expr::ModuleRef(
		                                                                            instance_target.clone(),
		                                                                            v.clone(),
		                                                                            Vec::new(),
		                                                                        ),
		                                                                        span: unch_inner.span,
		                                                                    },
		                                                                    _ => (**unch_inner).clone(),
		                                                                };

                                                                    let qualified_unchanged = Spanned {
		                                                                    node: Expr::Unchanged(Box::new(qualified_unch_inner)),
		                                                                    span: unchanged.span,
		                                                                };

                                                                    let (new_left, new_right) =
                                                                        if action_on_left {
                                                                            (
                                                                                qualified_action,
                                                                                qualified_unchanged,
                                                                            )
                                                                        } else {
                                                                            (
                                                                                qualified_unchanged,
                                                                                qualified_action,
                                                                            )
                                                                        };

                                                                    Some(Spanned {
                                                                        node: Expr::Or(
                                                                            Box::new(new_left),
                                                                            Box::new(new_right),
                                                                        ),
                                                                        span: inner.span,
                                                                    })
                                                                } else {
                                                                    None
                                                                }
                                                            }
                                                            // If the body is a bare identifier, qualify it into the instance scope.
                                                            Expr::Ident(action_name) => {
                                                                Some(Spanned {
                                                                    node: Expr::ModuleRef(
                                                                        instance_target.clone(),
                                                                        action_name.clone(),
                                                                        Vec::new(),
                                                                    ),
                                                                    span: inner.span,
                                                                })
                                                            }
                                                            _ => None,
                                                        };

                                                    if let Some(action_expr) = extracted_action {
                                                        always_terms.push(action_expr);
                                                        extracted_any = true;
                                                    } else {
                                                        liveness_terms.push(part.clone());
                                                    }
                                                }
                                                // Fairness/liveness terms stay in the liveness portion, but
                                                // rewrite WF/SF arguments to refer to instance actions.
                                                Expr::Apply(op, a) if a.len() == 1 => {
                                                    if let Expr::Ident(fname) = &op.node {
                                                        if fname.starts_with("WF_")
                                                            || fname.starts_with("SF_")
                                                        {
                                                            let arg = &a[0];
                                                            let new_arg = match &arg.node {
                                                                Expr::Ident(action_name) => {
                                                                    Spanned {
                                                                        node: Expr::ModuleRef(
                                                                            instance_target.clone(),
                                                                            action_name.clone(),
                                                                            Vec::new(),
                                                                        ),
                                                                        span: arg.span,
                                                                    }
                                                                }
                                                                _ => arg.clone(),
                                                            };
                                                            liveness_terms.push(Spanned {
                                                                node: Expr::Apply(
                                                                    op.clone(),
                                                                    vec![new_arg],
                                                                ),
                                                                span: part.span,
                                                            });
                                                            extracted_any = true;
                                                        } else {
                                                            liveness_terms.push(part.clone());
                                                        }
                                                    } else {
                                                        liveness_terms.push(part.clone());
                                                    }
                                                }
                                                Expr::WeakFair(sub, action)
                                                | Expr::StrongFair(sub, action) => {
                                                    let new_sub = match &sub.node {
                                                        Expr::Ident(v) => Spanned {
                                                            node: Expr::ModuleRef(
                                                                instance_target.clone(),
                                                                v.clone(),
                                                                Vec::new(),
                                                            ),
                                                            span: sub.span,
                                                        },
                                                        _ => (**sub).clone(),
                                                    };
                                                    let new_action = match &action.node {
                                                        Expr::Ident(a) => Spanned {
                                                            node: Expr::ModuleRef(
                                                                instance_target.clone(),
                                                                a.clone(),
                                                                Vec::new(),
                                                            ),
                                                            span: action.span,
                                                        },
                                                        _ => (**action).clone(),
                                                    };
                                                    liveness_terms.push(Spanned {
                                                        node: match &part.node {
                                                            Expr::WeakFair(_, _) => Expr::WeakFair(
                                                                Box::new(new_sub),
                                                                Box::new(new_action),
                                                            ),
                                                            _ => Expr::StrongFair(
                                                                Box::new(new_sub),
                                                                Box::new(new_action),
                                                            ),
                                                        },
                                                        span: part.span,
                                                    });
                                                    extracted_any = true;
                                                }
                                                // Nested ModuleRef: we cannot determine if the referenced
                                                // operator is temporal without recursively resolving it.
                                                // The nested ref needs to be evaluated in the outer
                                                // instance context, so push the ORIGINAL ModuleRef
                                                // (not the nested one) to liveness.
                                                Expr::ModuleRef(..) => {
                                                    // Push the original module ref to liveness, not the
                                                    // nested one - the liveness converter can resolve the
                                                    // full chain through `instance_target!op_name`.
                                                    liveness_terms.push(Spanned {
                                                        node: Expr::ModuleRef(
                                                            instance_target.clone(),
                                                            op_name.clone(),
                                                            Vec::new(),
                                                        ),
                                                        span: term.span,
                                                    });
                                                    extracted_any = true;
                                                }
                                                _ => {
                                                    // Default: treat as liveness unless it can be proven non-temporal.
                                                    if contains_temporal_local(&part.node)
                                                        || contains_enabled_local(&part.node)
                                                    {
                                                        liveness_terms.push(part.clone());
                                                    } else {
                                                        init_terms.push(part.clone());
                                                        extracted_any = true;
                                                    }
                                                }
                                            }
                                        }

                                        if extracted_any {
                                            // We handled this ModuleRef by splitting its body; skip the default liveness routing.
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    liveness_terms.push(term.clone());
                }
                // Non-temporal term - goes to init
                _ => {
                    if Self::contains_temporal_helper(&term.node)
                        || contains_enabled_local(&term.node)
                    {
                        liveness_terms.push(term.clone());
                    } else {
                        init_terms.push(term.clone());
                    }
                }
            }
        }

        // Build liveness expression as conjunction of liveness terms
        let liveness_expr = if liveness_terms.is_empty() {
            None
        } else if liveness_terms.len() == 1 {
            Some(liveness_terms.into_iter().next().unwrap())
        } else {
            // Build conjunction: term1 /\ term2 /\ ...
            let mut iter = liveness_terms.into_iter();
            let mut result = iter.next().unwrap();
            for term in iter {
                result = Spanned {
                    node: Expr::And(Box::new(result), Box::new(term)),
                    span: body.span,
                };
            }
            Some(result)
        };

        Some((
            PropertySafetyParts {
                init_terms,
                always_terms,
            },
            liveness_expr,
        ))
    }

    /// Helper to check if expression contains temporal operators
    fn contains_temporal_helper(expr: &Expr) -> bool {
        match expr {
            Expr::Always(_)
            | Expr::Eventually(_)
            | Expr::LeadsTo(_, _)
            | Expr::WeakFair(_, _)
            | Expr::StrongFair(_, _) => true,

            Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::Ident(_) | Expr::OpRef(_) => {
                false
            }

            // ENABLED is state-level but check its body
            Expr::Enabled(e) => Self::contains_temporal_helper(&e.node),

            Expr::Apply(op, args) => {
                // Check for WF_xxx/SF_xxx
                if let Expr::Ident(name) = &op.node {
                    if name.starts_with("WF_") || name.starts_with("SF_") {
                        return true;
                    }
                }
                Self::contains_temporal_helper(&op.node)
                    || args.iter().any(|a| Self::contains_temporal_helper(&a.node))
            }
            Expr::ModuleRef(_, _, args) => {
                args.iter().any(|a| Self::contains_temporal_helper(&a.node))
            }
            Expr::InstanceExpr(_, subs) => subs
                .iter()
                .any(|s| Self::contains_temporal_helper(&s.to.node)),
            Expr::Lambda(_params, body) => Self::contains_temporal_helper(&body.node),

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
            | Expr::Range(l, r) => {
                Self::contains_temporal_helper(&l.node) || Self::contains_temporal_helper(&r.node)
            }

            Expr::Not(e)
            | Expr::Powerset(e)
            | Expr::BigUnion(e)
            | Expr::Domain(e)
            | Expr::Prime(e)
            | Expr::Unchanged(e)
            | Expr::Neg(e) => Self::contains_temporal_helper(&e.node),

            Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
                bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .is_some_and(|d| Self::contains_temporal_helper(&d.node))
                }) || Self::contains_temporal_helper(&body.node)
            }
            Expr::Choose(bv, body) => {
                bv.domain
                    .as_ref()
                    .is_some_and(|d| Self::contains_temporal_helper(&d.node))
                    || Self::contains_temporal_helper(&body.node)
            }

            Expr::SetEnum(elems) => elems
                .iter()
                .any(|e| Self::contains_temporal_helper(&e.node)),
            Expr::SetBuilder(expr, bounds) => {
                Self::contains_temporal_helper(&expr.node)
                    || bounds.iter().any(|b| {
                        b.domain
                            .as_ref()
                            .is_some_and(|d| Self::contains_temporal_helper(&d.node))
                    })
            }
            Expr::SetFilter(bound, pred) => {
                bound
                    .domain
                    .as_ref()
                    .is_some_and(|d| Self::contains_temporal_helper(&d.node))
                    || Self::contains_temporal_helper(&pred.node)
            }

            Expr::FuncDef(bounds, body) => {
                bounds.iter().any(|b| {
                    b.domain
                        .as_ref()
                        .is_some_and(|d| Self::contains_temporal_helper(&d.node))
                }) || Self::contains_temporal_helper(&body.node)
            }
            Expr::FuncApply(f, arg) => {
                Self::contains_temporal_helper(&f.node) || Self::contains_temporal_helper(&arg.node)
            }
            Expr::Except(base, specs) => {
                Self::contains_temporal_helper(&base.node)
                    || specs
                        .iter()
                        .any(|s| Self::contains_temporal_helper(&s.value.node))
            }
            Expr::FuncSet(dom, ran) => {
                Self::contains_temporal_helper(&dom.node)
                    || Self::contains_temporal_helper(&ran.node)
            }

            Expr::Record(fields) => fields
                .iter()
                .any(|(_, v)| Self::contains_temporal_helper(&v.node)),
            Expr::RecordAccess(rec, _field) => Self::contains_temporal_helper(&rec.node),
            Expr::RecordSet(fields) => fields
                .iter()
                .any(|(_, v)| Self::contains_temporal_helper(&v.node)),

            Expr::Tuple(elems) | Expr::Times(elems) => elems
                .iter()
                .any(|e| Self::contains_temporal_helper(&e.node)),

            Expr::If(cond, then_e, else_e) => {
                Self::contains_temporal_helper(&cond.node)
                    || Self::contains_temporal_helper(&then_e.node)
                    || Self::contains_temporal_helper(&else_e.node)
            }
            Expr::Case(arms, other) => {
                arms.iter().any(|a| {
                    Self::contains_temporal_helper(&a.guard.node)
                        || Self::contains_temporal_helper(&a.body.node)
                }) || other
                    .as_ref()
                    .is_some_and(|e| Self::contains_temporal_helper(&e.node))
            }
            Expr::Let(defs, body) => {
                defs.iter()
                    .any(|d| Self::contains_temporal_helper(&d.body.node))
                    || Self::contains_temporal_helper(&body.node)
            }
        }
    }

    /// Check the safety parts of a property (init terms and always terms)
    ///
    /// Returns Some(CheckResult) if property is violated, None if satisfied.
    ///
    /// Under symmetry, `succ_witnesses` contains the actual concrete successors that were
    /// generated during exploration. This is important because `state_successors` contains
    /// representative states from `seen`, which may be different symmetric variants than
    /// the actual successors.
    fn check_property_safety_parts(
        &mut self,
        prop_name: &str,
        parts: &PropertySafetyParts,
        init_states: &[State],
        state_successors: &FxHashMap<Fingerprint, Vec<State>>,
        state_fp_to_canon_fp: &Arc<FxHashMap<Fingerprint, Fingerprint>>,
        succ_witnesses: Option<&Arc<SuccessorWitnessMap>>,
    ) -> Option<CheckResult> {
        if std::env::var("TLA2_DEBUG_SAFETY_PARTS").is_ok() {
            eprintln!("=== check_property_safety_parts called ===");
            eprintln!("  prop_name: {}", prop_name);
            eprintln!("  init_terms.len(): {}", parts.init_terms.len());
            eprintln!("  always_terms.len(): {}", parts.always_terms.len());
            eprintln!("  succ_witnesses.is_some(): {}", succ_witnesses.is_some());
        }
        // Check init terms on initial states
        for state in init_states {
            let saved_env = self.ctx.save_scope();
            let saved_next = self.ctx.next_state.clone();
            self.ctx.next_state = None;
            for (name, value) in state.vars() {
                self.ctx.bind_mut(Arc::clone(name), value.clone());
            }

            for term in &parts.init_terms {
                let v = match crate::eval::eval(&self.ctx, term) {
                    Ok(v) => v,
                    Err(e) => {
                        self.ctx.restore_scope(saved_env);
                        self.ctx.next_state = saved_next;
                        return Some(CheckResult::Error {
                            error: CheckError::EvalError(e),
                            stats: self.stats.clone(),
                        });
                    }
                };
                match v {
                    Value::Bool(true) => {}
                    Value::Bool(false) => {
                        let trace = Trace::from_states(vec![state.clone()]);
                        self.ctx.restore_scope(saved_env);
                        self.ctx.next_state = saved_next;
                        return Some(CheckResult::PropertyViolation {
                            property: prop_name.to_string(),
                            trace,
                            stats: self.stats.clone(),
                        });
                    }
                    _ => {
                        self.ctx.restore_scope(saved_env);
                        self.ctx.next_state = saved_next;
                        return Some(CheckResult::Error {
                            error: CheckError::PropertyNotBoolean(prop_name.to_string()),
                            stats: self.stats.clone(),
                        });
                    }
                }
            }

            self.ctx.restore_scope(saved_env);
            self.ctx.next_state = saved_next;
        }

        // Check always terms on transitions
        if !parts.always_terms.is_empty() {
            let registry = self.ctx.var_registry().clone();

            // Under symmetry, use succ_witnesses which contains actual concrete successors.
            // Without symmetry (or when succ_witnesses is None), use state_successors which
            // contains representative states.
            if let Some(witnesses) = succ_witnesses {
                if std::env::var("TLA2_DEBUG_SAFETY_PARTS").is_ok() {
                    eprintln!(
                        "=== check_property_safety_parts: using succ_witnesses (symmetry path) ==="
                    );
                    eprintln!("  witnesses.len() = {}", witnesses.len());
                }
                for (from_fp, witness_list) in witnesses.iter() {
                    // Reconstruct from_state from seen array states
                    let from_state = match self.seen.get(from_fp) {
                        Some(arr) => arr.to_state(&registry),
                        None => continue,
                    };

                    for (dest_canon_fp, to_state) in witness_list {
                        // Skip stuttering transitions (same canonical fp)
                        if from_fp == dest_canon_fp {
                            continue;
                        }

                        let saved_env = self.ctx.save_scope();
                        let saved_next = self.ctx.next_state.clone();

                        // Bind current state
                        for (name, value) in from_state.vars() {
                            self.ctx.bind_mut(Arc::clone(name), value.clone());
                        }
                        // Bind next state as environment for primed variable access
                        let mut next_env = crate::eval::Env::new();
                        for (name, value) in to_state.vars() {
                            next_env.insert(Arc::clone(name), value.clone());
                        }
                        self.ctx.next_state = Some(Arc::new(next_env));

                        for term in &parts.always_terms {
                            let v = match crate::eval::eval(&self.ctx, term) {
                                Ok(v) => v,
                                Err(e) => {
                                    self.ctx.restore_scope(saved_env);
                                    self.ctx.next_state = saved_next;
                                    return Some(CheckResult::Error {
                                        error: CheckError::EvalError(e),
                                        stats: self.stats.clone(),
                                    });
                                }
                            };
                            match v {
                                Value::Bool(true) => {}
                                Value::Bool(false) => {
                                    let trace = Trace::from_states(vec![
                                        from_state.clone(),
                                        to_state.clone(),
                                    ]);
                                    self.ctx.restore_scope(saved_env);
                                    self.ctx.next_state = saved_next;
                                    return Some(CheckResult::PropertyViolation {
                                        property: prop_name.to_string(),
                                        trace,
                                        stats: self.stats.clone(),
                                    });
                                }
                                _ => {
                                    self.ctx.restore_scope(saved_env);
                                    self.ctx.next_state = saved_next;
                                    return Some(CheckResult::Error {
                                        error: CheckError::PropertyNotBoolean(
                                            prop_name.to_string(),
                                        ),
                                        stats: self.stats.clone(),
                                    });
                                }
                            }
                        }

                        self.ctx.restore_scope(saved_env);
                        self.ctx.next_state = saved_next;
                    }
                }
            } else {
                // Non-symmetry path: use state_successors (representative states)
                if std::env::var("TLA2_DEBUG_SAFETY_PARTS").is_ok() {
                    eprintln!("=== check_property_safety_parts: using state_successors (non-symmetry path) ===");
                    eprintln!("  state_successors.len() = {}", state_successors.len());
                }
                for (from_fp, succs) in state_successors {
                    // Reconstruct from_state from seen array states
                    let from_state = match self.seen.get(from_fp) {
                        Some(arr) => arr.to_state(&registry),
                        None => continue,
                    };

                    for to_state in succs {
                        let saved_env = self.ctx.save_scope();
                        let saved_next = self.ctx.next_state.clone();

                        // Bind current state
                        for (name, value) in from_state.vars() {
                            self.ctx.bind_mut(Arc::clone(name), value.clone());
                        }
                        // Bind next state as environment for primed variable access
                        let mut next_env = crate::eval::Env::new();
                        for (name, value) in to_state.vars() {
                            next_env.insert(Arc::clone(name), value.clone());
                        }
                        self.ctx.next_state = Some(Arc::new(next_env));

                        for term in &parts.always_terms {
                            let v = match crate::eval::eval(&self.ctx, term) {
                                Ok(v) => v,
                                Err(e) => {
                                    self.ctx.restore_scope(saved_env);
                                    self.ctx.next_state = saved_next;
                                    return Some(CheckResult::Error {
                                        error: CheckError::EvalError(e),
                                        stats: self.stats.clone(),
                                    });
                                }
                            };
                            match v {
                                Value::Bool(true) => {}
                                Value::Bool(false) => {
                                    // Check if this is a stuttering step (from == to)
                                    // Under symmetry, compare canonical fingerprints - two states
                                    // in the same equivalence class count as stuttering.
                                    let from_fp_raw = from_state.fingerprint();
                                    let to_fp = to_state.fingerprint();
                                    let is_stuttering = if state_fp_to_canon_fp.is_empty() {
                                        from_fp_raw == to_fp
                                    } else {
                                        let canon_from =
                                            state_fp_to_canon_fp.get(&from_fp_raw).copied();
                                        let canon_to = state_fp_to_canon_fp.get(&to_fp).copied();
                                        match (canon_from, canon_to) {
                                            (Some(c1), Some(c2)) => c1 == c2,
                                            _ => from_fp_raw == to_fp,
                                        }
                                    };
                                    if is_stuttering {
                                        // Stuttering is always allowed in TLA+
                                        continue;
                                    }
                                    let trace = Trace::from_states(vec![
                                        from_state.clone(),
                                        to_state.clone(),
                                    ]);
                                    self.ctx.restore_scope(saved_env);
                                    self.ctx.next_state = saved_next;
                                    return Some(CheckResult::PropertyViolation {
                                        property: prop_name.to_string(),
                                        trace,
                                        stats: self.stats.clone(),
                                    });
                                }
                                _ => {
                                    self.ctx.restore_scope(saved_env);
                                    self.ctx.next_state = saved_next;
                                    return Some(CheckResult::Error {
                                        error: CheckError::PropertyNotBoolean(
                                            prop_name.to_string(),
                                        ),
                                        stats: self.stats.clone(),
                                    });
                                }
                            }
                        }

                        self.ctx.restore_scope(saved_env);
                        self.ctx.next_state = saved_next;
                    }
                }
            }
        }

        None
    }

    fn check_safety_temporal_property(
        &mut self,
        prop_name: &str,
        init_states: &[State],
        state_successors: &FxHashMap<Fingerprint, Vec<State>>,
        _state_fp_to_canon_fp: &Arc<FxHashMap<Fingerprint, Fingerprint>>,
        succ_witnesses: Option<&Arc<SuccessorWitnessMap>>,
    ) -> SafetyTemporalPropertyOutcome {
        fn flatten_and_terms(expr: &Spanned<Expr>, out: &mut Vec<Spanned<Expr>>) {
            match &expr.node {
                Expr::And(left, right) => {
                    flatten_and_terms(left, out);
                    flatten_and_terms(right, out);
                }
                _ => out.push(expr.clone()),
            }
        }

        fn contains_temporal(expr: &Expr) -> bool {
            match expr {
                Expr::Always(_)
                | Expr::Eventually(_)
                | Expr::LeadsTo(_, _)
                | Expr::WeakFair(_, _)
                | Expr::StrongFair(_, _)
                | Expr::Enabled(_) => true,

                Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) => false,

                // Identifiers may reference operators containing temporal formulas.
                // Conservatively return true to fall through to the liveness checker
                // which properly resolves operator references.
                Expr::Ident(_) => true,

                Expr::Apply(op, args) => {
                    contains_temporal(&op.node) || args.iter().any(|a| contains_temporal(&a.node))
                }
                Expr::ModuleRef(_, _, args) => args.iter().any(|a| contains_temporal(&a.node)),
                Expr::InstanceExpr(_, subs) => subs.iter().any(|s| contains_temporal(&s.to.node)),
                Expr::Lambda(_params, body) => contains_temporal(&body.node),

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
                | Expr::Range(l, r) => contains_temporal(&l.node) || contains_temporal(&r.node),

                Expr::Not(e)
                | Expr::Powerset(e)
                | Expr::BigUnion(e)
                | Expr::Domain(e)
                | Expr::Prime(e)
                | Expr::Unchanged(e)
                | Expr::Neg(e) => contains_temporal(&e.node),

                Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
                    bounds.iter().any(|b| {
                        b.domain
                            .as_ref()
                            .is_some_and(|d| contains_temporal(&d.node))
                    }) || contains_temporal(&body.node)
                }
                Expr::Choose(bv, body) => {
                    bv.domain
                        .as_ref()
                        .is_some_and(|d| contains_temporal(&d.node))
                        || contains_temporal(&body.node)
                }

                Expr::SetEnum(elems) => elems.iter().any(|e| contains_temporal(&e.node)),
                Expr::SetBuilder(expr, bounds) => {
                    contains_temporal(&expr.node)
                        || bounds.iter().any(|b| {
                            b.domain
                                .as_ref()
                                .is_some_and(|d| contains_temporal(&d.node))
                        })
                }
                Expr::SetFilter(bound, pred) => {
                    bound
                        .domain
                        .as_ref()
                        .is_some_and(|d| contains_temporal(&d.node))
                        || contains_temporal(&pred.node)
                }

                Expr::FuncDef(bounds, body) => {
                    bounds.iter().any(|b| {
                        b.domain
                            .as_ref()
                            .is_some_and(|d| contains_temporal(&d.node))
                    }) || contains_temporal(&body.node)
                }
                Expr::FuncApply(f, arg) => {
                    contains_temporal(&f.node) || contains_temporal(&arg.node)
                }
                Expr::Except(base, specs) => {
                    contains_temporal(&base.node)
                        || specs.iter().any(|s| contains_temporal(&s.value.node))
                }
                Expr::FuncSet(dom, ran) => {
                    contains_temporal(&dom.node) || contains_temporal(&ran.node)
                }

                Expr::Record(fields) => fields.iter().any(|(_, v)| contains_temporal(&v.node)),
                Expr::RecordAccess(rec, _field) => contains_temporal(&rec.node),
                Expr::RecordSet(fields) => fields.iter().any(|(_, v)| contains_temporal(&v.node)),

                Expr::Tuple(elems) | Expr::Times(elems) => {
                    elems.iter().any(|e| contains_temporal(&e.node))
                }

                Expr::If(cond, then_e, else_e) => {
                    contains_temporal(&cond.node)
                        || contains_temporal(&then_e.node)
                        || contains_temporal(&else_e.node)
                }
                Expr::Case(arms, other) => {
                    arms.iter().any(|a| {
                        contains_temporal(&a.guard.node) || contains_temporal(&a.body.node)
                    }) || other.as_ref().is_some_and(|e| contains_temporal(&e.node))
                }
                Expr::Let(defs, body) => {
                    defs.iter().any(|d| contains_temporal(&d.body.node))
                        || contains_temporal(&body.node)
                }
            }
        }

        // Only attempt the safety check for properties that are conjunctions of:
        // - state-level terms (checked on initial states), and
        // - [] terms whose bodies contain no temporal operators (checked on transitions).
        let def = match self.op_defs.get(prop_name) {
            Some(d) => d.clone(),
            None => {
                return SafetyTemporalPropertyOutcome::Violated(Box::new(CheckResult::Error {
                    error: CheckError::MissingProperty(prop_name.to_string()),
                    stats: self.stats.clone(),
                }));
            }
        };

        // Many specs define properties with a top-level LET (e.g., `TreeWithRoot` in EWD687a):
        //
        //   TreeWithRoot == LET ... IN []( ... )
        //
        // For the safety-temporal fast path, unwrap the LET for structural matching,
        // but keep the LET in scope when evaluating extracted terms.
        let (let_defs, prop_body) = match &def.body.node {
            Expr::Let(defs, body) => (Some(defs.clone()), (**body).clone()),
            _ => (None, def.body.clone()),
        };

        fn wrap_with_let(defs: &Option<Vec<OperatorDef>>, expr: Spanned<Expr>) -> Spanned<Expr> {
            match defs {
                Some(defs) => {
                    Spanned::new(Expr::Let(defs.clone(), Box::new(expr.clone())), expr.span)
                }
                None => expr,
            }
        }

        let mut terms = Vec::new();
        flatten_and_terms(&prop_body, &mut terms);

        let mut init_terms = Vec::new();
        let mut always_terms = Vec::new();

        for term in terms {
            match &term.node {
                Expr::Always(inner) => {
                    if contains_temporal(&inner.node) {
                        return SafetyTemporalPropertyOutcome::NotApplicable;
                    }
                    always_terms.push(wrap_with_let(&let_defs, (**inner).clone()));
                }
                _ => {
                    if contains_temporal(&term.node) {
                        return SafetyTemporalPropertyOutcome::NotApplicable;
                    }
                    init_terms.push(wrap_with_let(&let_defs, term));
                }
            }
        }

        if always_terms.is_empty() {
            return SafetyTemporalPropertyOutcome::NotApplicable;
        }

        let prop = SafetyTemporalProperty {
            init_terms,
            always_terms,
        };

        // Split `[]P` terms into:
        // - state-level predicates (P is state/constant): check on each reachable state
        // - action-level predicates (P is action): check on each explored transition
        //
        // This is critical for performance. Without this split, `[]StatePredicate` would be
        // redundantly evaluated once per transition (|E|) instead of once per state (|V|).
        let converter = crate::liveness::AstToLive::new();
        let mut always_state_terms: Vec<Spanned<Expr>> = Vec::new();
        let mut always_action_terms: Vec<Spanned<Expr>> = Vec::new();
        for term in &prop.always_terms {
            match converter.get_level_with_ctx(&self.ctx, &term.node) {
                crate::liveness::ExprLevel::Constant | crate::liveness::ExprLevel::State => {
                    always_state_terms.push(term.clone());
                }
                crate::liveness::ExprLevel::Action => {
                    always_action_terms.push(term.clone());
                }
                crate::liveness::ExprLevel::Temporal => {
                    return SafetyTemporalPropertyOutcome::NotApplicable;
                }
            }
        }

        // Check initial-state terms.
        for state in init_states {
            let saved_env = self.ctx.save_scope();
            let saved_next = self.ctx.next_state.clone();
            self.ctx.next_state = None;
            for (name, value) in state.vars() {
                self.ctx.bind_mut(Arc::clone(name), value.clone());
            }

            for term in &prop.init_terms {
                let v = match crate::eval::eval(&self.ctx, term) {
                    Ok(v) => v,
                    Err(e) => {
                        self.ctx.restore_scope(saved_env);
                        self.ctx.next_state = saved_next;
                        return SafetyTemporalPropertyOutcome::Violated(Box::new(
                            CheckResult::Error {
                                error: CheckError::EvalError(e),
                                stats: self.stats.clone(),
                            },
                        ));
                    }
                };
                match v {
                    Value::Bool(true) => {}
                    Value::Bool(false) => {
                        let trace = Trace::from_states(vec![state.clone()]);
                        self.ctx.restore_scope(saved_env);
                        self.ctx.next_state = saved_next;
                        return SafetyTemporalPropertyOutcome::Violated(Box::new(
                            CheckResult::PropertyViolation {
                                property: prop_name.to_string(),
                                trace,
                                stats: self.stats.clone(),
                            },
                        ));
                    }
                    _ => {
                        self.ctx.restore_scope(saved_env);
                        self.ctx.next_state = saved_next;
                        return SafetyTemporalPropertyOutcome::Violated(Box::new(
                            CheckResult::Error {
                                error: CheckError::PropertyNotBoolean(prop_name.to_string()),
                                stats: self.stats.clone(),
                            },
                        ));
                    }
                }
            }

            self.ctx.restore_scope(saved_env);
            self.ctx.next_state = saved_next;
        }

        // Check state-level [] terms on each reachable state.
        for fp in state_successors.keys() {
            let Some(cur_array) = self.seen.get(fp) else {
                continue;
            };

            let saved_next = self.ctx.next_state.clone();
            let prev_state_env = self.ctx.bind_state_array(cur_array.values());
            self.ctx.next_state = None;

            for term in &always_state_terms {
                let v = match crate::eval::eval(&self.ctx, term) {
                    Ok(v) => v,
                    Err(e) => {
                        self.ctx.restore_state_env(prev_state_env);
                        self.ctx.next_state = saved_next;
                        return SafetyTemporalPropertyOutcome::Violated(Box::new(
                            CheckResult::Error {
                                error: CheckError::EvalError(e),
                                stats: self.stats.clone(),
                            },
                        ));
                    }
                };

                match v {
                    Value::Bool(true) => {}
                    Value::Bool(false) => {
                        let trace = self.reconstruct_trace(*fp);
                        self.ctx.restore_state_env(prev_state_env);
                        self.ctx.next_state = saved_next;
                        return SafetyTemporalPropertyOutcome::Violated(Box::new(
                            CheckResult::PropertyViolation {
                                property: prop_name.to_string(),
                                trace,
                                stats: self.stats.clone(),
                            },
                        ));
                    }
                    _ => {
                        self.ctx.restore_state_env(prev_state_env);
                        self.ctx.next_state = saved_next;
                        return SafetyTemporalPropertyOutcome::Violated(Box::new(
                            CheckResult::Error {
                                error: CheckError::PropertyNotBoolean(prop_name.to_string()),
                                stats: self.stats.clone(),
                            },
                        ));
                    }
                }
            }

            self.ctx.restore_state_env(prev_state_env);
            self.ctx.next_state = saved_next;
        }

        // Check action-level [] terms on each explored transition.
        if always_action_terms.is_empty() {
            return SafetyTemporalPropertyOutcome::Satisfied;
        }

        let registry = self.ctx.var_registry().clone();

        // Under symmetry, use succ_witnesses which contains actual concrete successors.
        // state_successors contains REPRESENTATIVE states which may not be valid successors
        // of the source representative.
        if let Some(witnesses) = succ_witnesses {
            for (fp, witness_list) in witnesses.iter() {
                let Some(cur_array) = self.seen.get(fp) else {
                    continue;
                };
                let cur_state = cur_array.to_state(&registry);

                let saved_env = self.ctx.save_scope();
                let saved_next = self.ctx.next_state.clone();
                for (name, value) in cur_state.vars() {
                    self.ctx.bind_mut(Arc::clone(name), value.clone());
                }

                for (dest_canon_fp, succ_state) in witness_list {
                    // Skip stuttering transitions (same canonical fp)
                    if fp == dest_canon_fp {
                        continue;
                    }

                    let mut next_env = Env::new();
                    for (name, value) in succ_state.vars() {
                        next_env.insert(Arc::clone(name), value.clone());
                    }
                    self.ctx.next_state = Some(std::sync::Arc::new(next_env));

                    for term in &always_action_terms {
                        let v = match crate::eval::eval(&self.ctx, term) {
                            Ok(v) => v,
                            Err(e) => {
                                self.ctx.restore_scope(saved_env);
                                self.ctx.next_state = saved_next;
                                return SafetyTemporalPropertyOutcome::Violated(Box::new(
                                    CheckResult::Error {
                                        error: CheckError::EvalError(e),
                                        stats: self.stats.clone(),
                                    },
                                ));
                            }
                        };
                        match v {
                            Value::Bool(true) => {}
                            Value::Bool(false) => {
                                // DEBUG: Print the actual states involved in the violation
                                if std::env::var("TLA2_DEBUG_SAFETY_TEMPORAL").is_ok() {
                                    eprintln!("=== Safety Temporal Property Violation ===");
                                    eprintln!("Source canon fp: {}", fp);
                                    eprintln!("Dest canon fp: {}", dest_canon_fp);
                                    eprintln!("Current state (from seen):");
                                    for (name, value) in cur_state.vars() {
                                        eprintln!("  {} = {:?}", name, value);
                                    }
                                    eprintln!("Successor state (from witness):");
                                    for (name, value) in succ_state.vars() {
                                        eprintln!("  {} = {:?}", name, value);
                                    }
                                    eprintln!("==========================================");
                                }
                                let succ_fp = self.state_fingerprint(succ_state);
                                let trace = self.reconstruct_trace(succ_fp);
                                self.ctx.restore_scope(saved_env);
                                self.ctx.next_state = saved_next;
                                return SafetyTemporalPropertyOutcome::Violated(Box::new(
                                    CheckResult::PropertyViolation {
                                        property: prop_name.to_string(),
                                        trace,
                                        stats: self.stats.clone(),
                                    },
                                ));
                            }
                            _ => {
                                self.ctx.restore_scope(saved_env);
                                self.ctx.next_state = saved_next;
                                return SafetyTemporalPropertyOutcome::Violated(Box::new(
                                    CheckResult::Error {
                                        error: CheckError::PropertyNotBoolean(
                                            prop_name.to_string(),
                                        ),
                                        stats: self.stats.clone(),
                                    },
                                ));
                            }
                        }
                    }
                }

                self.ctx.restore_scope(saved_env);
                self.ctx.next_state = saved_next;
            }
        } else {
            // No symmetry - use state_successors directly
            for (fp, succ_states) in state_successors {
                let Some(cur_array) = self.seen.get(fp) else {
                    continue;
                };
                let cur_state = cur_array.to_state(&registry);

                let saved_env = self.ctx.save_scope();
                let saved_next = self.ctx.next_state.clone();
                for (name, value) in cur_state.vars() {
                    self.ctx.bind_mut(Arc::clone(name), value.clone());
                }

                for succ_state in succ_states {
                    let succ_fp = self.state_fingerprint(succ_state);
                    // Skip stuttering transitions
                    if *fp == succ_fp {
                        continue;
                    }

                    let mut next_env = Env::new();
                    for (name, value) in succ_state.vars() {
                        next_env.insert(Arc::clone(name), value.clone());
                    }
                    self.ctx.next_state = Some(std::sync::Arc::new(next_env));

                    for term in &always_action_terms {
                        let v = match crate::eval::eval(&self.ctx, term) {
                            Ok(v) => v,
                            Err(e) => {
                                self.ctx.restore_scope(saved_env);
                                self.ctx.next_state = saved_next;
                                return SafetyTemporalPropertyOutcome::Violated(Box::new(
                                    CheckResult::Error {
                                        error: CheckError::EvalError(e),
                                        stats: self.stats.clone(),
                                    },
                                ));
                            }
                        };
                        match v {
                            Value::Bool(true) => {}
                            Value::Bool(false) => {
                                let trace = self.reconstruct_trace(succ_fp);
                                self.ctx.restore_scope(saved_env);
                                self.ctx.next_state = saved_next;
                                return SafetyTemporalPropertyOutcome::Violated(Box::new(
                                    CheckResult::PropertyViolation {
                                        property: prop_name.to_string(),
                                        trace,
                                        stats: self.stats.clone(),
                                    },
                                ));
                            }
                            _ => {
                                self.ctx.restore_scope(saved_env);
                                self.ctx.next_state = saved_next;
                                return SafetyTemporalPropertyOutcome::Violated(Box::new(
                                    CheckResult::Error {
                                        error: CheckError::PropertyNotBoolean(
                                            prop_name.to_string(),
                                        ),
                                        stats: self.stats.clone(),
                                    },
                                ));
                            }
                        }
                    }
                }

                self.ctx.restore_scope(saved_env);
                self.ctx.next_state = saved_next;
            }
        }

        SafetyTemporalPropertyOutcome::Satisfied
    }

    // =========================================================================
    // CHECKPOINT/RESUME SUPPORT
    // =========================================================================

    /// Create a checkpoint of the current model checking state.
    ///
    /// This captures all data needed to resume model checking later:
    /// - Seen fingerprints
    /// - Frontier states (from the provided queue)
    /// - Parent pointers
    /// - Depth tracking
    /// - Statistics
    ///
    /// # Arguments
    /// * `frontier` - The current BFS frontier (states to explore)
    /// * `spec_path` - Optional spec file path for verification on resume
    /// * `config_path` - Optional config file path for verification on resume
    pub fn create_checkpoint(
        &self,
        frontier: &std::collections::VecDeque<State>,
        spec_path: Option<&str>,
        config_path: Option<&str>,
    ) -> crate::checkpoint::Checkpoint {
        use crate::checkpoint::{Checkpoint, CheckpointStats};

        let mut checkpoint = Checkpoint::new().with_paths(spec_path, config_path);

        // Collect fingerprints from depths keys.
        //
        // We intentionally avoid relying on iterating the fingerprint set because:
        // - `MmapFingerprintSet` doesn't support iteration
        // - After resume, we may have fingerprints restored without full states
        checkpoint.fingerprints = self.depths.keys().copied().collect();

        // Copy frontier states
        checkpoint.frontier = frontier.iter().cloned().collect();

        // Copy parent pointers
        checkpoint.parents = self.parents.iter().map(|(k, v)| (*k, *v)).collect();

        // Copy depths
        checkpoint.depths = self.depths.iter().map(|(k, v)| (*k, *v)).collect();

        // Set stats
        checkpoint.metadata.stats = CheckpointStats::from(&self.stats);
        checkpoint.metadata.stats.frontier_size = frontier.len();

        checkpoint
    }

    /// Restore model checking state from a checkpoint.
    ///
    /// This restores:
    /// - Seen fingerprints to the fingerprint set
    /// - Parent pointers
    /// - Depth tracking
    /// - Statistics
    ///
    /// The frontier is returned separately so the caller can resume BFS.
    ///
    /// # Returns
    /// The frontier states to continue BFS from
    pub fn restore_from_checkpoint(
        &mut self,
        checkpoint: crate::checkpoint::Checkpoint,
    ) -> std::collections::VecDeque<State> {
        // Clear in-memory state maps (checkpoint restores these).
        self.seen.clear();
        self.parents.clear();
        self.depths.clear();
        self.cached_successors.clear();
        self.cached_successor_states.clear();

        // Restore fingerprints to the seen set
        for fp in &checkpoint.fingerprints {
            // Mark as seen in the fingerprint set
            self.seen_fps.insert(*fp);

            // If we have full states mode, we don't have the actual states
            // So we can't populate self.seen - trace reconstruction will need
            // to use fingerprint replay
        }

        // Restore parent pointers
        self.parents = checkpoint.parents.into_iter().collect();

        // Restore depths
        self.depths = checkpoint.depths.into_iter().collect();

        // Restore statistics
        self.stats.states_found = checkpoint.metadata.stats.states_found;
        self.stats.initial_states = checkpoint.metadata.stats.initial_states;
        self.stats.transitions = checkpoint.metadata.stats.transitions;
        self.stats.max_depth = checkpoint.metadata.stats.max_depth;

        // If configured to store full states, keep the frontier states in memory.
        // This improves trace quality immediately after resume without requiring replay.
        if self.store_full_states {
            let registry = self.ctx.var_registry().clone();
            for state in &checkpoint.frontier {
                let fp = self.state_fingerprint(state);
                // Convert State to ArrayState for storage
                let array_state = ArrayState::from_state(state, &registry);
                self.seen.insert(fp, array_state);
            }
        }

        // Return frontier for caller to resume BFS
        checkpoint.frontier.into_iter().collect()
    }

    /// Check if this checker can be checkpointed.
    ///
    /// Returns true if the checker configuration supports checkpointing.
    /// Currently, checkpointing works best with store_full_states=true mode.
    pub fn supports_checkpoint(&self) -> bool {
        // Checkpointing is supported but works best with full state storage
        // Memory-mapped fingerprint sets don't support iteration for checkpoint
        true
    }
}

/// Simple model checking entry point
pub fn check_module(module: &Module, config: &Config) -> CheckResult {
    let mut checker = ModelChecker::new(module, config);
    checker.check()
}

// =============================================================================
// SIMULATION MODE
// =============================================================================

/// Configuration for simulation mode
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Number of random traces to generate
    pub num_traces: usize,
    /// Maximum length of each trace (steps from initial state)
    pub max_trace_length: usize,
    /// Random seed for reproducibility (None = random seed)
    pub seed: Option<u64>,
    /// Whether to check invariants during simulation
    pub check_invariants: bool,
    /// Action constraints from config
    pub action_constraints: Vec<String>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            num_traces: 1000,
            max_trace_length: 100,
            seed: None,
            check_invariants: true,
            action_constraints: Vec::new(),
        }
    }
}

/// Statistics from simulation
#[derive(Debug, Clone, Default)]
pub struct SimulationStats {
    /// Total number of traces generated
    pub traces_generated: usize,
    /// Total number of states visited (may include duplicates across traces)
    pub states_visited: usize,
    /// Number of distinct states seen
    pub distinct_states: usize,
    /// Number of transitions taken
    pub transitions: usize,
    /// Maximum trace length achieved
    pub max_trace_length: usize,
    /// Average trace length
    pub avg_trace_length: f64,
    /// Number of traces that hit deadlock
    pub deadlocked_traces: usize,
    /// Number of traces that hit max length limit
    pub truncated_traces: usize,
    /// Elapsed time in seconds
    pub elapsed_secs: f64,
}

/// Result of simulation
#[derive(Debug, Clone)]
pub enum SimulationResult {
    /// Simulation completed without finding violations
    Success(SimulationStats),
    /// An invariant was violated during simulation
    InvariantViolation {
        invariant: String,
        trace: Trace,
        stats: SimulationStats,
    },
    /// Error during simulation
    Error {
        error: CheckError,
        stats: SimulationStats,
    },
}

/// A simple linear congruential generator for reproducible randomness
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng {
            state: seed.wrapping_add(1),
        }
    }

    fn from_entropy() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        SimpleRng::new(seed)
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % (bound as u64)) as usize
    }
}

impl ModelChecker<'_> {
    /// Run simulation mode - generate random traces through the state space
    ///
    /// Simulation mode is useful for:
    /// - Quick exploration of large state spaces
    /// - Finding bugs that require deep traces
    /// - Probabilistic coverage when exhaustive checking is infeasible
    ///
    /// Unlike exhaustive model checking, simulation does not guarantee complete
    /// coverage but can explore much deeper into the state space.
    pub fn simulate(&mut self, sim_config: &SimulationConfig) -> SimulationResult {
        let start_time = Instant::now();

        // Sync TLC config for TLCGet("config") support - use "generate" mode for simulation
        self.sync_tlc_config("generate");

        // Bind constants from config before checking
        if let Err(e) = bind_constants_from_config(&mut self.ctx, self.config) {
            return SimulationResult::Error {
                error: CheckError::EvalError(e),
                stats: SimulationStats::default(),
            };
        }

        // Validate config
        let init_name = match &self.config.init {
            Some(name) => name.clone(),
            None => {
                return SimulationResult::Error {
                    error: CheckError::MissingInit,
                    stats: SimulationStats::default(),
                }
            }
        };

        let next_name = match &self.config.next {
            Some(name) => name.clone(),
            None => {
                return SimulationResult::Error {
                    error: CheckError::MissingNext,
                    stats: SimulationStats::default(),
                }
            }
        };

        if self.vars.is_empty() {
            return SimulationResult::Error {
                error: CheckError::NoVariables,
                stats: SimulationStats::default(),
            };
        }

        // Validate invariants exist
        for inv_name in &self.config.invariants {
            if !self.ctx.has_op(inv_name) {
                return SimulationResult::Error {
                    error: CheckError::MissingInvariant(inv_name.clone()),
                    stats: SimulationStats::default(),
                };
            }
        }

        // Generate initial states
        let initial_states = match self.generate_initial_states(&init_name) {
            Ok(states) => states,
            Err(e) => {
                return SimulationResult::Error {
                    error: e,
                    stats: SimulationStats::default(),
                }
            }
        };

        // Filter initial states by state constraints
        let initial_states: Vec<State> = initial_states
            .into_iter()
            .filter(|state| self.check_state_constraints(state))
            .collect();

        if initial_states.is_empty() {
            return SimulationResult::Error {
                error: CheckError::InitCannotEnumerate(
                    "No valid initial states after constraint filtering".to_string(),
                ),
                stats: SimulationStats::default(),
            };
        }

        // Initialize RNG
        let mut rng = match sim_config.seed {
            Some(seed) => SimpleRng::new(seed),
            None => SimpleRng::from_entropy(),
        };

        let mut stats = SimulationStats::default();
        let mut seen: FxHashSet<Fingerprint> = FxHashSet::default();
        let mut total_trace_length: usize = 0;

        // Generate random traces
        for _trace_num in 0..sim_config.num_traces {
            // Pick a random initial state
            let init_idx = rng.next_usize(initial_states.len());
            let mut current_state = initial_states[init_idx].clone();
            let mut trace: Vec<State> = vec![current_state.clone()];

            // Record state
            let fp = current_state.fingerprint();
            seen.insert(fp);
            stats.states_visited += 1;

            // Check invariants on initial state
            if sim_config.check_invariants {
                if let Some(violation) = self.check_invariants(&current_state) {
                    stats.traces_generated = _trace_num + 1;
                    stats.distinct_states = seen.len();
                    stats.elapsed_secs = start_time.elapsed().as_secs_f64();
                    return SimulationResult::InvariantViolation {
                        invariant: violation,
                        trace: Trace::from_states(trace),
                        stats,
                    };
                }
            }

            // Random walk
            let mut trace_length = 0;
            loop {
                if trace_length >= sim_config.max_trace_length {
                    stats.truncated_traces += 1;
                    break;
                }

                // Generate successors
                let succ_result =
                    match self.generate_successors_filtered(&next_name, &current_state) {
                        Ok(result) => result,
                        Err(e) => {
                            stats.traces_generated = _trace_num + 1;
                            stats.distinct_states = seen.len();
                            stats.elapsed_secs = start_time.elapsed().as_secs_f64();
                            return SimulationResult::Error { error: e, stats };
                        }
                    };
                let valid_successors = succ_result.successors;

                if valid_successors.is_empty() {
                    // Deadlock - no valid successors (only if there were no raw successors)
                    // Note: For simulation, we count states with no successors as deadlocks
                    // even if they were filtered by constraints. This matches behavior of
                    // random exploration hitting constraint boundaries.
                    stats.deadlocked_traces += 1;
                    break;
                }

                // Pick a random successor
                let succ_idx = rng.next_usize(valid_successors.len());
                let next_state = valid_successors.into_iter().nth(succ_idx).unwrap();

                stats.transitions += 1;
                trace_length += 1;

                // Record state
                let fp = next_state.fingerprint();
                seen.insert(fp);
                stats.states_visited += 1;

                // Check invariants
                if sim_config.check_invariants {
                    if let Some(violation) = self.check_invariants(&next_state) {
                        trace.push(next_state);
                        stats.traces_generated = _trace_num + 1;
                        stats.distinct_states = seen.len();
                        stats.elapsed_secs = start_time.elapsed().as_secs_f64();
                        return SimulationResult::InvariantViolation {
                            invariant: violation,
                            trace: Trace::from_states(trace),
                            stats,
                        };
                    }
                }

                trace.push(next_state.clone());
                current_state = next_state;
            }

            total_trace_length += trace_length;
            stats.max_trace_length = stats.max_trace_length.max(trace_length);
            stats.traces_generated += 1;
        }

        stats.distinct_states = seen.len();
        stats.avg_trace_length = if stats.traces_generated > 0 {
            total_trace_length as f64 / stats.traces_generated as f64
        } else {
            0.0
        };
        stats.elapsed_secs = start_time.elapsed().as_secs_f64();

        SimulationResult::Success(stats)
    }
}

/// Simple simulation entry point
pub fn simulate_module(
    module: &Module,
    config: &Config,
    sim_config: &SimulationConfig,
) -> SimulationResult {
    let mut checker = ModelChecker::new(module, config);
    checker.simulate(sim_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_display() {
        let s1 = State::from_pairs([("x", Value::int(0))]);
        let s2 = State::from_pairs([("x", Value::int(1))]);
        let trace = Trace::from_states(vec![s1, s2]);

        let display = format!("{}", trace);
        assert!(display.contains("State 1"));
        assert!(display.contains("State 2"));
        assert!(display.contains("x = 0"));
        assert!(display.contains("x = 1"));
    }

    #[test]
    fn test_check_stats_default() {
        let stats = CheckStats::default();
        assert_eq!(stats.states_found, 0);
        assert_eq!(stats.initial_states, 0);
        assert_eq!(stats.transitions, 0);
    }

    #[test]
    fn test_check_error_display() {
        let err = CheckError::MissingInit;
        assert_eq!(format!("{}", err), "Missing INIT definition");

        let err = CheckError::MissingInvariant("Safety".to_string());
        assert!(format!("{}", err).contains("Safety"));
    }

    #[test]
    fn test_check_missing_init() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        let src = r#"
---- MODULE Test ----
VARIABLE x
Next == x' = x + 1
TypeOK == x \in Nat
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config without INIT
        let config = Config {
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        let result = check_module(&module, &config);
        assert!(matches!(
            result,
            CheckResult::Error {
                error: CheckError::MissingInit,
                ..
            }
        ));
    }

    #[test]
    fn test_check_missing_next() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
TypeOK == x \in Nat
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config without NEXT
        let config = Config {
            init: Some("Init".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        let result = check_module(&module, &config);
        assert!(matches!(
            result,
            CheckResult::Error {
                error: CheckError::MissingNext,
                ..
            }
        ));
    }

    #[test]
    fn test_check_assume_only_no_init_next() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        let src = r#"
---- MODULE Test ----
ASSUME TRUE
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config without INIT/NEXT/SPECIFICATION, no variables, no properties/invariants.
        let config = Config::default();

        let result = check_module(&module, &config);
        assert!(matches!(result, CheckResult::Success(_)));
    }

    #[test]
    fn test_check_missing_invariant() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config with non-existent invariant
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["NonExistent".to_string()],
            ..Default::default()
        };

        let result = check_module(&module, &config);
        assert!(matches!(
            result,
            CheckResult::Error { error: CheckError::MissingInvariant(ref name), .. } if name == "NonExistent"
        ));
    }

    #[test]
    fn test_check_init_cannot_enumerate() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Init constrains x to an infinite set (Nat), which we cannot enumerate
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x \in Nat
Next == x' = x + 1
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        let result = check_module(&module, &config);
        // This should fail because Nat is an infinite set that cannot be enumerated
        assert!(matches!(
            result,
            CheckResult::Error {
                error: CheckError::InitCannotEnumerate(_),
                ..
            }
        ));
    }

    #[test]
    fn test_check_init_missing_variable_constraint() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Init only constrains one of two variables
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x = 0
Next == x' = x + 1 /\ y' = y
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        let result = check_module(&module, &config);
        // This should fail because y has no constraint
        assert!(matches!(
            result,
            CheckResult::Error {
                error: CheckError::InitCannotEnumerate(ref msg),
                ..
            } if msg.contains("y")
        ));
    }

    #[test]
    fn test_model_checker_construction() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        let src = r#"
---- MODULE Counter ----
VARIABLE count
Init == count = 0
Next == count' = count + 1
TypeOK == count \in Nat
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        let checker = ModelChecker::new(&module, &config);

        // Verify variable extraction
        assert_eq!(checker.vars.len(), 1);
        assert_eq!(checker.vars[0].as_ref(), "count");
    }

    // ============================
    // End-to-end model checking tests
    // ============================

    #[test]
    fn test_model_check_simple_counter() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // A counter that increments from 0 to 2 with a bounded constraint
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
InRange == x >= 0 /\ x <= 2
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["InRange".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        // Allow deadlock since x=2 has no successors (x < 2 is false)
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // Should find 3 states: x=0, x=1, x=2
                assert_eq!(stats.states_found, 3);
                assert_eq!(stats.initial_states, 1);
                // Transitions: 0->1, 1->2 = 2 transitions
                assert_eq!(stats.transitions, 2);
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    #[test]
    fn test_model_check_invariant_violation() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Counter starts at 0, increments unboundedly
        // Invariant x < 3 should be violated when x reaches 3
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
SmallValue == x < 3
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["SmallValue".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::InvariantViolation {
                invariant,
                trace,
                stats,
            } => {
                assert_eq!(invariant, "SmallValue");
                // Trace should show path from x=0 to x=3
                assert_eq!(trace.len(), 4); // x=0, x=1, x=2, x=3
                assert!(stats.states_found >= 3);
            }
            other => panic!("Expected invariant violation, got: {:?}", other),
        }
    }

    #[test]
    fn test_model_check_multiple_initial_states() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Multiple initial states: x can be 0 or 1
        let src = r#"
---- MODULE Multi ----
VARIABLE x
Init == x \in {0, 1}
Next == x < 3 /\ x' = x + 1
InRange == x >= 0 /\ x <= 3
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["InRange".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // Initial: x=0, x=1
                // From x=0: can reach x=1, x=2, x=3
                // From x=1: can reach x=2, x=3
                // Unique states: 0, 1, 2, 3 = 4 states
                assert_eq!(stats.initial_states, 2);
                assert_eq!(stats.states_found, 4);
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    #[test]
    fn test_model_check_disjunctive_init() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Disjunctive Init: x starts as 0 or 1
        let src = r#"
---- MODULE DisjInit ----
VARIABLE x
Init == x = 0 \/ x = 1
Next == x < 3 /\ x' = x + 1
InRange == x >= 0 /\ x <= 3
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["InRange".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.initial_states, 2);
                assert_eq!(stats.states_found, 4); // x=0,1,2,3
                assert_eq!(stats.transitions, 3);
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    #[test]
    fn test_model_check_two_variables() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Two variables: x increments, y stays the same
        let src = r#"
---- MODULE TwoVars ----
VARIABLE x, y
Init == x = 0 /\ y = 5
Next == x' = x + 1 /\ UNCHANGED y
Bounded == x < 2
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Bounded".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::InvariantViolation {
                invariant,
                trace,
                stats: _,
            } => {
                assert_eq!(invariant, "Bounded");
                // Should find violation at x=2, y=5
                assert!(trace.len() >= 3); // x=0, x=1, x=2

                // Verify y stayed unchanged in all states
                for state in &trace.states {
                    let y_val = state.vars().find(|(n, _)| n.as_ref() == "y");
                    assert!(y_val.is_some());
                    assert_eq!(y_val.unwrap().1, &Value::int(5));
                }
            }
            other => panic!("Expected invariant violation, got: {:?}", other),
        }
    }

    #[test]
    fn test_model_check_disjunctive_next() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Next has disjunction: can increment by 1 or 2
        let src = r#"
---- MODULE Disjunction ----
VARIABLE x
Init == x = 0
Next == x' = x + 1 \/ x' = x + 2
SmallValue == x < 4
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["SmallValue".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::InvariantViolation {
                invariant,
                trace,
                stats: _,
            } => {
                assert_eq!(invariant, "SmallValue");
                // Can reach x=4 in 2 steps (0->2->4) or 4 steps (0->1->2->3->4)
                assert!(trace.len() >= 3);
            }
            other => panic!("Expected invariant violation, got: {:?}", other),
        }
    }

    #[test]
    fn test_model_check_in_set_next() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test x' \in S in Next relation - counter can jump by 1, 2, or 3
        let src = r#"
---- MODULE InSetCounter ----
VARIABLE x
Init == x = 0
Next == x' \in {x + 1, x + 2, x + 3}
SmallValue == x < 5
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["SmallValue".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::InvariantViolation {
                invariant,
                trace,
                stats,
            } => {
                assert_eq!(invariant, "SmallValue");
                // Can reach x=5 in as few as 2 steps (0->3->6) or 0->2->5
                assert!(trace.len() >= 2);
                // Should have explored states 0,1,2,3,4 (at least 5)
                assert!(stats.states_found >= 5);
            }
            other => panic!("Expected invariant violation, got: {:?}", other),
        }
    }

    #[test]
    fn test_model_check_in_set_cartesian() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test x' \in S and y' \in T - cartesian product of transitions
        let src = r#"
---- MODULE Cartesian ----
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == x' \in {0, 1} /\ y' \in {0, 1}
Sum == x + y < 2
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Sum".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::InvariantViolation {
                invariant,
                trace,
                stats,
            } => {
                assert_eq!(invariant, "Sum");
                // x=1, y=1 violates Sum (1+1 = 2, not < 2)
                // Total states: (0,0), (0,1), (1,0), (1,1) = 4 states
                assert!(stats.states_found >= 3); // At least some states before violation
                assert!(trace.len() >= 2);
            }
            other => panic!("Expected invariant violation, got: {:?}", other),
        }
    }

    // ============================
    // Exploration limit tests
    // ============================

    #[test]
    fn test_max_states_limit() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Unbounded counter that would run forever without limits
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_max_states(5);

        let result = checker.check();

        match result {
            CheckResult::LimitReached { limit_type, stats } => {
                assert_eq!(limit_type, LimitType::States);
                assert_eq!(stats.states_found, 5);
            }
            other => panic!("Expected LimitReached(States), got: {:?}", other),
        }
    }

    #[test]
    fn test_max_depth_limit() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Unbounded counter that would run forever without limits
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_max_depth(3);

        let result = checker.check();

        match result {
            CheckResult::LimitReached { limit_type, stats } => {
                assert_eq!(limit_type, LimitType::Depth);
                // Should have explored: x=0 (depth 0), x=1 (depth 1), x=2 (depth 2), x=3 (depth 3)
                // But x=3 is at max_depth so we don't generate x=4
                assert_eq!(stats.states_found, 4);
                assert_eq!(stats.max_depth, 3);
            }
            other => panic!("Expected LimitReached(Depth), got: {:?}", other),
        }
    }

    #[test]
    fn test_invariant_found_before_limit() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Counter with invariant that will be violated before hitting limit
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
SmallValue == x < 3
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["SmallValue".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_max_states(100); // High limit that won't be reached

        let result = checker.check();

        match result {
            CheckResult::InvariantViolation {
                invariant, stats, ..
            } => {
                assert_eq!(invariant, "SmallValue");
                // Should find violation at x=3 before hitting 100 state limit
                assert!(stats.states_found < 100);
            }
            other => panic!("Expected InvariantViolation, got: {:?}", other),
        }
    }

    #[test]
    fn test_success_within_limits() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Bounded counter that terminates naturally
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x < 3 /\ x' = x + 1
InRange == x >= 0 /\ x <= 3
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["InRange".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_max_states(100);
        checker.set_max_depth(100);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // Should complete naturally with 4 states
                assert_eq!(stats.states_found, 4);
            }
            other => panic!("Expected Success, got: {:?}", other),
        }
    }

    #[test]
    fn test_depth_tracking() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Bounded counter
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x < 5 /\ x' = x + 1
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 6); // x=0,1,2,3,4,5
                assert_eq!(stats.max_depth, 5); // 0->1->2->3->4->5
            }
            other => panic!("Expected Success, got: {:?}", other),
        }
    }

    // ============================
    // Progress callback tests
    // ============================

    #[test]
    fn test_progress_callback() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Counter that explores a moderate number of states
        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x < 100 /\ x' = x + 1
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_clone = Arc::clone(&callback_count);

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_progress_interval(10); // Report every 10 states
        checker.set_progress_callback(Box::new(move |progress| {
            callback_count_clone.fetch_add(1, Ordering::SeqCst);
            // Verify progress values are reasonable
            assert!(progress.states_found > 0);
        }));

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 101); // x=0,1,2,...,100
                                                     // Should have been called approximately 10 times (101 states / 10 interval)
                let count = callback_count.load(Ordering::SeqCst);
                assert!(
                    count >= 5,
                    "Expected at least 5 progress callbacks, got {}",
                    count
                );
            }
            other => panic!("Expected Success, got: {:?}", other),
        }
    }

    #[test]
    fn test_progress_callback_disabled() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        let src = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x < 10 /\ x' = x + 1
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_clone = Arc::clone(&callback_count);

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_progress_interval(0); // Disabled
        checker.set_progress_callback(Box::new(move |_| {
            callback_count_clone.fetch_add(1, Ordering::SeqCst);
        }));

        let result = checker.check();

        match result {
            CheckResult::Success(_) => {
                // Callback should never be called when interval is 0
                let count = callback_count.load(Ordering::SeqCst);
                assert_eq!(
                    count, 0,
                    "Expected 0 callbacks when disabled, got {}",
                    count
                );
            }
            other => panic!("Expected Success, got: {:?}", other),
        }
    }

    // ============================
    // SPECIFICATION directive tests
    // ============================

    #[test]
    fn test_resolve_spec_explicit_init_next() {
        use tla_core::parse_to_syntax_tree;

        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
===="#;
        let tree = parse_to_syntax_tree(src);

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let resolved = super::resolve_spec_from_config(&config, &tree).unwrap();
        assert_eq!(resolved.init, "Init");
        assert_eq!(resolved.next, "Next");
    }

    #[test]
    fn test_resolve_spec_from_specification() {
        use tla_core::parse_to_syntax_tree;

        let src = r#"
---- MODULE Test ----
VARIABLE x
vars == <<x>>
Init == x = 0
Next == x' = x + 1
Spec == Init /\ [][Next]_vars
===="#;
        let tree = parse_to_syntax_tree(src);

        let config = Config {
            specification: Some("Spec".to_string()),
            ..Default::default()
        };

        let resolved = super::resolve_spec_from_config(&config, &tree).unwrap();
        assert_eq!(resolved.init, "Init");
        assert_eq!(resolved.next, "Next");
    }

    #[test]
    fn test_resolve_spec_from_specification_with_wrapper_operator_in_extends() {
        use tla_core::parse_to_syntax_tree;

        let base = r#"
---- MODULE Base ----
VARIABLE x
vars == <<x>>
Init == x = 0
Next == x' = x + 1
Spec == Init /\ [][Next]_vars
===="#;
        let base_tree = parse_to_syntax_tree(base);

        let main = r#"
---- MODULE Main ----
EXTENDS Base
Foo == TRUE
TestSpec == Foo /\ Spec
===="#;
        let main_tree = parse_to_syntax_tree(main);

        let config = Config {
            specification: Some("TestSpec".to_string()),
            ..Default::default()
        };

        let resolved =
            super::resolve_spec_from_config_with_extends(&config, &main_tree, &[&base_tree])
                .unwrap();
        assert_eq!(resolved.init, "Init");
        assert_eq!(resolved.next, "Next");
    }

    #[test]
    fn test_resolve_spec_missing_specification() {
        use tla_core::parse_to_syntax_tree;

        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
===="#;
        let tree = parse_to_syntax_tree(src);

        let config = Config {
            specification: Some("NonExistent".to_string()),
            ..Default::default()
        };

        let result = super::resolve_spec_from_config(&config, &tree);
        assert!(result.is_err());
        if let Err(CheckError::SpecificationError(msg)) = result {
            assert!(msg.contains("NonExistent"));
        } else {
            panic!("Expected SpecificationError");
        }
    }

    #[test]
    fn test_resolve_spec_missing_init_next() {
        use tla_core::parse_to_syntax_tree;

        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
===="#;
        let tree = parse_to_syntax_tree(src);

        let config = Config::default();

        let result = super::resolve_spec_from_config(&config, &tree);
        assert!(result.is_err());
        assert!(matches!(result, Err(CheckError::MissingInit)));
    }

    #[test]
    fn test_resolve_spec_inline_next_expression() {
        use tla_core::parse_to_syntax_tree;

        // Test case: Spec with inline existential quantifier in NEXT
        let src = r#"
---- MODULE Test ----
CONSTANT Node
VARIABLE x
vars == x
Init == x = 0
Next(n) == x' = n
Spec == Init /\ [][\E n \in Node: Next(n)]_vars
===="#;
        let tree = parse_to_syntax_tree(src);

        let config = Config {
            specification: Some("Spec".to_string()),
            ..Default::default()
        };

        let resolved = super::resolve_spec_from_config(&config, &tree).unwrap();
        assert_eq!(resolved.init, "Init");
        // When there's an inline NEXT, we use the synthetic name
        assert_eq!(resolved.next, super::INLINE_NEXT_NAME);
        // The next_node should be present
        assert!(resolved.next_node.is_some());
    }

    #[test]
    fn test_model_checker_with_inline_next() {
        use crate::config::ConstantValue;
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // A simple spec using inline existential in NEXT
        let src = r#"
---- MODULE InlineNextTest ----
EXTENDS Integers
CONSTANT Node
VARIABLE x
vars == <<x>>
Init == x = 0
Step(n) == x < 3 /\ x' = x + 1
Spec == Init /\ [][\E n \in Node: Step(n)]_vars
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let spec_config = Config {
            specification: Some("Spec".to_string()),
            ..Default::default()
        };
        let resolved = super::resolve_spec_from_config(&spec_config, &tree).unwrap();

        // Create checker config with the resolved NEXT name
        let mut constants = std::collections::HashMap::new();
        constants.insert("Node".to_string(), ConstantValue::Value("{1}".to_string()));

        let config = Config {
            init: Some(resolved.init.clone()),
            next: Some(resolved.next.clone()),
            constants,
            invariants: vec![],
            ..Default::default()
        };

        let mut checker = super::ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        // Register the inline NEXT expression
        checker
            .register_inline_next(&resolved)
            .expect("Failed to register inline next");

        let result = checker.check();

        // Should succeed and find states: x=0, x=1, x=2, x=3
        match result {
            super::CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 4, "Expected 4 states (x=0,1,2,3)");
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    // ============================
    // CONSTANTS binding tests
    // ============================

    #[test]
    fn test_model_check_with_integer_constant() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Spec with CONSTANT N used in Init and invariant
        let src = r#"
---- MODULE ConstTest ----
CONSTANT N
VARIABLE x
Init == x = N
Next == x' = x + 1
Bounded == x < N + 5
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config with N = 3
        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Bounded".to_string()],
            ..Default::default()
        };
        config.constants.insert(
            "N".to_string(),
            crate::config::ConstantValue::Value("3".to_string()),
        );

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::InvariantViolation {
                invariant,
                trace,
                stats: _,
            } => {
                assert_eq!(invariant, "Bounded");
                // N=3, starts at x=3, invariant x < 8 (3+5)
                // Violation at x=8
                // Trace: x=3, x=4, x=5, x=6, x=7, x=8
                assert_eq!(trace.len(), 6);
            }
            other => panic!("Expected invariant violation, got: {:?}", other),
        }
    }

    #[test]
    fn test_model_check_with_model_value_set_constant() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Spec with CONSTANT Procs as model value set
        let src = r#"
---- MODULE SetConstTest ----
CONSTANT Procs
VARIABLE current
Init == current \in Procs
Next == current' \in Procs
AlwaysProc == current \in Procs
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config with Procs = {p1, p2}
        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["AlwaysProc".to_string()],
            ..Default::default()
        };
        config.constants.insert(
            "Procs".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
        );

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // Should find 2 states: current=p1, current=p2
                assert_eq!(stats.states_found, 2);
                assert_eq!(stats.initial_states, 2);
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    #[test]
    fn test_model_value_equality_in_invariant() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test that model value equality works correctly in compiled invariants
        // This tests the fix for the bug where `decision = none` would fail
        // when `none` is a model value constant (like in SimplifiedFastPaxos)
        let src = r#"
---- MODULE ModelValueEqTest ----
CONSTANT none, Values
VARIABLE decision, messages

Init ==
    /\ decision = none
    /\ messages = {}

Next ==
    /\ messages' = messages \union {[value |-> CHOOSE v \in Values : TRUE]}
    /\ UNCHANGED decision

\* This invariant should always be true since decision never changes from none
DecisionIsNone == decision = none
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["DecisionIsNone".to_string()],
            ..Default::default()
        };
        // none = none is a model value assignment
        config.constants.insert(
            "none".to_string(),
            crate::config::ConstantValue::Value("none".to_string()),
        );
        // Values = {v1}
        config.constants.insert(
            "Values".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec!["v1".to_string()]),
        );

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // Should find 2 states: initial (empty messages) and one with a message
                assert_eq!(stats.states_found, 2);
            }
            other => panic!(
                "Expected success (model value equality should work), got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_func_set_membership_in_invariant() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test function set membership in invariant (like Barrier's TypeOK)
        let src = r#"
---- MODULE FuncTest ----
CONSTANT N
VARIABLE pc
ProcSet == 1..N
Init == pc = [p \in ProcSet |-> "b0"]
Next == UNCHANGED pc
TypeOK == pc \in [ProcSet -> {"b0", "b1"}]
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config with N = 3 (smaller for faster test)
        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };
        config.constants.insert(
            "N".to_string(),
            crate::config::ConstantValue::Value("3".to_string()),
        );

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // Should find 1 state with pc = [1 |-> "b0", 2 |-> "b0", 3 |-> "b0"]
                assert_eq!(stats.states_found, 1);
                assert_eq!(stats.initial_states, 1);
            }
            other => panic!("Expected success (TypeOK should pass), got: {:?}", other),
        }
    }

    #[test]
    fn test_let_guard_evaluation() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test that guards inside LET expressions are properly evaluated.
        // This is a simplified version of the MissionariesAndCannibals spec pattern:
        //   Move(items) == LET newThis == ... IN /\ IsSafe(newThis) /\ x' = ...
        // The IsSafe guard inside the LET body must be checked before generating successors.
        let spec = r#"
---- MODULE LetGuard ----
EXTENDS Integers, FiniteSets

VARIABLES x, y

Init == x = {1, 2} /\ y = {}

IsSafe(S) == Cardinality(S) <= 1

Move(items) ==
    /\ items \subseteq x
    /\ Cardinality(items) = 1
    /\ LET newX == x \ items
           newY == y \cup items
       IN  /\ IsSafe(newX)
           /\ x' = newX
           /\ y' = newY

Next == \E items \in SUBSET x : Move(items)

TypeOK == x \subseteq {1, 2} /\ y \subseteq {1, 2}

====
"#;
        let tree = parse_to_syntax_tree(spec);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // From Init: x={1,2}, y={}
                // Move {1}: newX={2}, IsSafe({2})=TRUE -> x={2}, y={1}
                // Move {2}: newX={1}, IsSafe({1})=TRUE -> x={1}, y={2}
                // From x={2}: Move {2}: newX={}, IsSafe({})=TRUE -> x={}, y={1,2}
                // From x={1}: Move {1}: newX={}, IsSafe({})=TRUE -> x={}, y={1,2}
                // Both paths converge to x={}, y={1,2}
                // Total distinct states: {1,2}/{}; {2}/{1}; {1}/{2}; {}/{1,2} = 4 states
                assert_eq!(stats.states_found, 4);
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    #[test]
    fn test_selectseq_guard_evaluation() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test that guards using SelectSeq properly limit the state space
        // This uses the exact pattern from ReadersWriters.tla:
        // - SelectSeq filters the waiting queue by type
        // - Guard prevents actors from re-entering the queue
        let src = r#"
---- MODULE SelectSeqGuard ----
EXTENDS Sequences
CONSTANT Actors
VARIABLE waiting

ToSet(s) == { s[i] : i \in DOMAIN s }

\* Predicate to filter by type
is_read(p) == p[1] = "read"

\* Get set of actor IDs with "read" requests (using SelectSeq)
WaitingToRead == { p[2] : p \in ToSet(SelectSeq(waiting, is_read)) }

Init == waiting = <<>>

\* Actor can only request read if not already waiting
TryRead(actor) ==
    /\ actor \notin WaitingToRead
    /\ waiting' = Append(waiting, <<"read", actor>>)

Next == \E actor \in Actors : TryRead(actor)
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config: Actors = {1, 2}
        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };
        config.constants.insert(
            "Actors".to_string(),
            crate::config::ConstantValue::Value("{1, 2}".to_string()),
        );

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_max_states(100); // Safety limit

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // From Init: waiting=<<>>
                // 2 actors can request: <<("request", 1)>> or <<("request", 2)>>
                // After one requests, the other can still request
                // Final states: <<>>, <<("read",1)>>, <<("read",2)>>,
                //               <<("read",1),("read",2)>>, <<("read",2),("read",1)>>
                // No actor can request twice because of the guard.
                // Total: 5 distinct states
                // Note: Without the guard, the queue would grow unboundedly.
                assert_eq!(
                    stats.states_found, 5,
                    "State space should be exactly 5 with guard (found {})",
                    stats.states_found
                );
            }
            CheckResult::LimitReached {
                limit_type: _,
                stats,
            } => {
                panic!(
                    "Guard failed to limit state space! Found {} states before limit",
                    stats.states_found
                );
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    #[test]
    fn test_inlined_operator_substitution_avoids_variable_capture() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Regression test: compiled inlining substitutes operator parameters into the operator body.
        // This must avoid variable capture when the argument identifier matches a locally-bound name
        // inside the operator body (here: `RemovePending(c)` where `RemovePending` defines `filter(c)`).
        let src = r#"
---- MODULE CaptureAvoidingInline ----
EXTENDS Sequences
VARIABLE pending

RemovePending(cmd) ==
    LET filter(c) == c # cmd
    IN  SelectSeq(pending, filter)

Init == pending = <<1, 2>>

Next == LET c == 1 IN pending' = RemovePending(c)

NotEmpty == Len(pending) > 0
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["NotEmpty".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_max_states(16);

        let result = checker.check();
        match result {
            CheckResult::Success(stats) => {
                // pending: <<1,2>> -> <<2>> (then stutters on <<2>>)
                assert_eq!(stats.states_found, 2);
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    #[test]
    fn test_case_expression_in_next_action() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test CASE expression handling in Next actions
        // This pattern appears in ReadersWriters.tla:
        //   CASE pair[1] = "read" -> Read(actor)
        //     [] pair[1] = "write" -> Write(actor)
        let src = r#"
---- MODULE CaseAction ----
VARIABLE state, value

Init == state = "start" /\ value = 0

ProcessRead ==
    /\ value' = value + 1
    /\ state' = "read"

ProcessWrite ==
    /\ value' = value + 10
    /\ state' = "write"

\* CASE-based action selection
DoAction(request) ==
    CASE request = "read" -> ProcessRead
      [] request = "write" -> ProcessWrite

Next ==
    \/ state = "start" /\ DoAction("read")
    \/ state = "start" /\ DoAction("write")
    \/ state = "read" /\ state' = "done" /\ UNCHANGED value
    \/ state = "write" /\ state' = "done" /\ UNCHANGED value

====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // States:
                // 1. (start, 0)     -- initial
                // 2. (read, 1)      -- from DoAction("read")
                // 3. (write, 10)    -- from DoAction("write")
                // 4. (done, 1)      -- from read -> done
                // 5. (done, 10)     -- from write -> done
                assert_eq!(
                    stats.states_found, 5,
                    "Should find 5 states with CASE-based actions (found {})",
                    stats.states_found
                );
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    #[test]
    fn test_state_constraints() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test CONSTRAINT directive support
        // Constraint limits state space by filtering states that don't satisfy it
        let src = r#"
---- MODULE ConstraintTest ----
EXTENDS Integers

VARIABLE x

Init == x = 0

Next == x' = x + 1

\* Without constraint, this would explore infinitely
\* With constraint x <= 5, only states 0-5 are explored
Constraint == x <= 5

TypeOK == x \in Nat
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // With constraint, should find exactly 6 states (x = 0, 1, 2, 3, 4, 5)
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            constraints: vec!["Constraint".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(
                    stats.states_found, 6,
                    "With constraint x <= 5, should find exactly 6 states (x = 0..5), found {}",
                    stats.states_found
                );
                assert_eq!(stats.initial_states, 1, "Should have 1 initial state");
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    #[test]
    fn test_safety_temporal_property_init_and_always_action_satisfied() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // PROPERTY that is actually a safety-style spec formula:
        //   Init /\ [][Next]_vars
        // should be checkable without the full liveness tableau algorithm.
        let src = r#"
---- MODULE SafetyPropSat ----
EXTENDS Integers

VARIABLE x
vars == <<x>>

Init == x = 0

Next == IF x < 2 THEN x' = x + 1 ELSE UNCHANGED x

SpecProp == Init /\ [][Next]_vars
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            properties: vec!["SpecProp".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();
        match result {
            CheckResult::Success(stats) => {
                // x takes values 0, 1, 2
                assert_eq!(stats.states_found, 3);
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    #[test]
    fn test_safety_temporal_property_init_and_always_action_violated() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Next increments, but property requires stuttering (UNCHANGED x) on every step.
        let src = r#"
---- MODULE SafetyPropViol ----
EXTENDS Integers

VARIABLE x
vars == <<x>>

Init == x = 0

Next == IF x < 2 THEN x' = x + 1 ELSE UNCHANGED x

Bad == UNCHANGED x
SpecProp == Init /\ [][Bad]_vars
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            properties: vec!["SpecProp".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_store_states(true); // Property checking requires full states

        let result = checker.check();
        match result {
            CheckResult::PropertyViolation {
                property,
                trace,
                stats: _,
            } => {
                assert_eq!(property, "SpecProp");
                assert!(trace.len() >= 2, "Expected at least 2 states in trace");
            }
            other => panic!("Expected property violation, got: {:?}", other),
        }
    }

    // ============================
    // Liveness checking tests
    // ============================

    #[test]
    fn test_temporal_enabled_counts_stuttering_successors() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Regression test: ENABLED(A) must consider stuttering successors (s -> s).
        // In particular, ENABLED(UNCHANGED x) should be TRUE, since UNCHANGED x is always enabled.
        let src = r#"
---- MODULE EnabledStutter ----
VARIABLE x
vars == <<x>>

Init == x = 0

Stutter == UNCHANGED x
Next == Stutter

EnabledStutter == ENABLED Stutter
Prop == []EnabledStutter
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            properties: vec!["Prop".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();
        assert!(
            matches!(result, CheckResult::Success(_)),
            "expected success, got: {result:?}"
        );
    }

    #[test]
    fn test_liveness_eventually_satisfied() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Spec where x increments from 0 to 2 then stops
        // Property <>P where P == x = 2 should be satisfied WITH fairness
        // Without fairness (WF), the system could stutter forever at x=0 or x=1
        let src = r#"
---- MODULE LivenessTest ----
EXTENDS Integers

VARIABLE x
vars == <<x>>

Init == x = 0

Inc == x < 2 /\ x' = x + 1

Next == Inc \/ UNCHANGED x

\* Spec with weak fairness - ensures Inc happens when enabled
Spec == Init /\ [][Next]_vars /\ WF_vars(Inc)

\* Eventually x reaches 2
EventuallyTwo == <>(x = 2)
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // First resolve the spec to get init/next/fairness
        let spec_config = Config {
            specification: Some("Spec".to_string()),
            ..Default::default()
        };
        let resolved = resolve_spec_from_config(&spec_config, &tree).unwrap();

        // Now create checker with explicit init/next
        let config = Config {
            init: Some(resolved.init.clone()),
            next: Some(resolved.next.clone()),
            invariants: vec![],
            properties: vec!["EventuallyTwo".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_fairness(resolved.fairness);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // Should find 3 states: x=0, x=1, x=2
                assert_eq!(stats.states_found, 3);
            }
            other => panic!("Expected success (liveness satisfied), got: {:?}", other),
        }
    }

    #[test]
    fn test_liveness_eventually_violated() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Spec where x stays at 0 forever
        // Property <>P where P == x = 1 should be violated
        let src = r#"
---- MODULE LivenessViolatedTest ----
EXTENDS Integers

VARIABLE x

Init == x = 0

\* x never changes
Next == UNCHANGED x

\* Eventually x reaches 1 - but it never does!
EventuallyOne == <>(x = 1)
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            properties: vec!["EventuallyOne".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_store_states(true); // Liveness checking requires full states

        let result = checker.check();

        match result {
            CheckResult::LivenessViolation {
                property,
                prefix: _,
                cycle,
                stats: _,
            } => {
                assert_eq!(property, "EventuallyOne");
                // The cycle should be non-empty (self-loop on x=0)
                assert!(!cycle.is_empty(), "Cycle should be non-empty");
            }
            other => panic!("Expected liveness violation, got: {:?}", other),
        }
    }

    #[test]
    fn test_liveness_always_eventually_violated() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // System that toggles between 0 and 1 forever
        // Property []<>(x = 2) should be violated because x never reaches 2
        let src = r#"
---- MODULE AlwaysEventuallyTest ----
EXTENDS Integers

VARIABLE x

Init == x = 0

\* Toggle between 0 and 1
Next == x' = IF x = 0 THEN 1 ELSE 0

\* Infinitely often x = 2 - but it never is!
InfOftenTwo == []<>(x = 2)
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            properties: vec!["InfOftenTwo".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_store_states(true); // Liveness checking requires full states

        let result = checker.check();

        match result {
            CheckResult::LivenessViolation {
                property,
                prefix: _,
                cycle,
                stats,
            } => {
                assert_eq!(property, "InfOftenTwo");
                // The cycle involves states 0 and 1
                assert!(!cycle.is_empty(), "Cycle should be non-empty");
                assert_eq!(stats.states_found, 2);
            }
            other => panic!("Expected liveness violation, got: {:?}", other),
        }
    }

    #[test]
    fn test_liveness_always_eventually_satisfied() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // System that toggles between 0 and 1 forever
        // Property []<>(x = 1) should be satisfied because x oscillates through 1
        // Requires fairness (WF) to prevent infinite stuttering at x=0
        let src = r#"
---- MODULE AlwaysEventuallySatTest ----
EXTENDS Integers

VARIABLE x
vars == <<x>>

Init == x = 0

\* Toggle between 0 and 1
Toggle == x' = IF x = 0 THEN 1 ELSE 0

Next == Toggle

\* Spec with weak fairness - ensures Toggle happens
Spec == Init /\ [][Next]_vars /\ WF_vars(Toggle)

\* Infinitely often x = 1 - satisfied with fairness!
InfOftenOne == []<>(x = 1)
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // First resolve the spec to get init/next/fairness
        let spec_config = Config {
            specification: Some("Spec".to_string()),
            ..Default::default()
        };
        let resolved = resolve_spec_from_config(&spec_config, &tree).unwrap();

        // Now create checker with explicit init/next
        let config = Config {
            init: Some(resolved.init.clone()),
            next: Some(resolved.next.clone()),
            invariants: vec![],
            properties: vec!["InfOftenOne".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_fairness(resolved.fairness);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 2);
            }
            other => panic!("Expected success (liveness satisfied), got: {:?}", other),
        }
    }

    #[test]
    fn test_fairness_extraction_from_spec() {
        use tla_core::parse_to_syntax_tree;

        // Test that fairness is extracted from SPECIFICATION formula
        let src = r#"
---- MODULE FairnessTest ----
VARIABLE x
vars == <<x>>
Init == x = 0
Next == x' = x + 1
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
===="#;
        let tree = parse_to_syntax_tree(src);

        let config = Config {
            specification: Some("Spec".to_string()),
            ..Default::default()
        };

        let resolved = super::resolve_spec_from_config(&config, &tree).unwrap();
        assert_eq!(resolved.init, "Init");
        assert_eq!(resolved.next, "Next");
        assert_eq!(resolved.fairness.len(), 1);

        match &resolved.fairness[0] {
            FairnessConstraint::Weak { vars, action, .. } => {
                assert_eq!(vars, "vars");
                assert_eq!(action, "Next");
            }
            _ => panic!("Expected weak fairness"),
        }
    }

    #[test]
    fn test_fairness_set_and_get() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        let src = r#"
---- MODULE SetFairnessTest ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x' = x + 1
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);

        // Initially no fairness
        assert!(checker.fairness.is_empty());

        // Set fairness
        let fairness = vec![FairnessConstraint::Weak {
            vars: "vars".to_string(),
            action: "Next".to_string(),
            action_node: None,
        }];
        checker.set_fairness(fairness);

        assert_eq!(checker.fairness.len(), 1);
    }

    #[test]
    fn test_fairness_extraction_from_nested_spec() {
        use tla_core::parse_to_syntax_tree;

        // Test that fairness is extracted from nested SPECIFICATION formula
        // SpecWeakFair == Spec /\ WF_vars(Next) where Spec contains Init/Next
        let src = r#"
---- MODULE NestedFairnessTest ----
VARIABLE x
vars == <<x>>
Init == x = 0
Next == x' = x + 1
Spec == Init /\ [][Next]_vars
SpecWeakFair == Spec /\ WF_vars(Next)
===="#;
        let tree = parse_to_syntax_tree(src);

        let config = Config {
            specification: Some("SpecWeakFair".to_string()),
            ..Default::default()
        };

        let resolved = super::resolve_spec_from_config(&config, &tree).unwrap();
        assert_eq!(resolved.init, "Init");
        assert_eq!(resolved.next, "Next");
        assert_eq!(
            resolved.fairness.len(),
            1,
            "Should have extracted fairness from nested spec"
        );

        match &resolved.fairness[0] {
            FairnessConstraint::Weak { vars, action, .. } => {
                assert_eq!(vars, "vars");
                assert_eq!(action, "Next");
            }
            _ => panic!("Expected weak fairness"),
        }
    }

    #[test]
    fn test_liveness_with_disjunctive_action_and_fairness() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // SystemLoop test case: disjunctive Next action with weak fairness
        // This is the Manna & Pneuli LOOP example
        let src = r#"
---- MODULE SystemLoopTest ----

VARIABLE x
vars == <<x>>

Init == x = 0

One == x = 0 /\ x' = 1
Two == x = 1 /\ x' = 2
Three == x = 2 /\ x' = 3
Back == x = 3 /\ x' = 0

Next == One \/ Two \/ Three \/ Back

\* Spec with weak fairness
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Liveness: x will infinitely often be 3
Liveness == []<>(x = 3)

===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Resolve spec to get init/next/fairness
        let spec_config = Config {
            specification: Some("Spec".to_string()),
            ..Default::default()
        };
        let resolved = resolve_spec_from_config(&spec_config, &tree).unwrap();

        // Check fairness was extracted
        assert_eq!(
            resolved.fairness.len(),
            1,
            "Should have weak fairness extracted"
        );

        // Create checker with explicit init/next and fairness
        let config = Config {
            init: Some(resolved.init.clone()),
            next: Some(resolved.next.clone()),
            invariants: vec![],
            properties: vec!["Liveness".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        checker.set_fairness(resolved.fairness);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // 4 states: x=0, x=1, x=2, x=3
                assert_eq!(stats.states_found, 4);
            }
            CheckResult::LivenessViolation { .. } => {
                panic!("Liveness should be satisfied with weak fairness (no stuttering)");
            }
            CheckResult::Error { error, .. } => {
                panic!("Unexpected error: {:?}", error);
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }

    // ==========================================================================
    // Simulation mode tests
    // ==========================================================================

    #[test]
    fn test_simulation_basic() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Simple counter that goes 0 -> 1 -> 2 -> 3 -> 4 -> 5 (deadlock)
        let src = r#"
---- MODULE SimTest ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 5 /\ x' = x + 1
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let sim_config = SimulationConfig {
            num_traces: 10,
            max_trace_length: 100,
            seed: Some(42),
            check_invariants: true,
            action_constraints: Vec::new(),
        };

        let mut checker = ModelChecker::new(&module, &config);
        let result = checker.simulate(&sim_config);

        match result {
            SimulationResult::Success(stats) => {
                assert_eq!(stats.traces_generated, 10);
                // All traces should deadlock at x=5
                assert_eq!(stats.deadlocked_traces, 10);
                assert_eq!(stats.truncated_traces, 0);
                // Should find all 6 distinct states (0..5)
                assert_eq!(stats.distinct_states, 6);
                // Max trace length is 5 (0 -> 1 -> 2 -> 3 -> 4 -> 5)
                assert_eq!(stats.max_trace_length, 5);
            }
            other => panic!("Expected Success, got {:?}", other),
        }
    }

    #[test]
    fn test_simulation_with_invariant_violation() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Counter that goes 0 -> 1 -> 2 -> 3, but invariant says x <= 2
        let src = r#"
---- MODULE SimInvariantTest ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 5 /\ x' = x + 1
SafetyInvariant == x <= 2
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["SafetyInvariant".to_string()],
            ..Default::default()
        };

        let sim_config = SimulationConfig {
            num_traces: 100,
            max_trace_length: 100,
            seed: Some(42),
            check_invariants: true,
            action_constraints: Vec::new(),
        };

        let mut checker = ModelChecker::new(&module, &config);
        let result = checker.simulate(&sim_config);

        match result {
            SimulationResult::InvariantViolation {
                invariant,
                trace,
                stats: _,
            } => {
                assert_eq!(invariant, "SafetyInvariant");
                // Trace should end at a state where x = 3 (violates x <= 2)
                let last_state = trace.states.last().unwrap();
                let x_val = last_state.get("x").unwrap();
                assert_eq!(*x_val, Value::int(3));
            }
            other => panic!("Expected InvariantViolation, got {:?}", other),
        }
    }

    #[test]
    fn test_simulation_reproducible_with_seed() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Branching spec with multiple choices at each step
        let src = r#"
---- MODULE SimSeedTest ----
EXTENDS Naturals
VARIABLE x
Init == x \in {1, 2, 3}
Next == x' \in {x + 1, x + 2, x + 3}
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        // Run twice with same seed
        let sim_config = SimulationConfig {
            num_traces: 10,
            max_trace_length: 5,
            seed: Some(12345),
            check_invariants: false,
            action_constraints: Vec::new(),
        };

        let mut checker1 = ModelChecker::new(&module, &config);
        let result1 = checker1.simulate(&sim_config);

        let mut checker2 = ModelChecker::new(&module, &config);
        let result2 = checker2.simulate(&sim_config);

        // Results should be identical
        match (result1, result2) {
            (SimulationResult::Success(stats1), SimulationResult::Success(stats2)) => {
                assert_eq!(stats1.traces_generated, stats2.traces_generated);
                assert_eq!(stats1.states_visited, stats2.states_visited);
                assert_eq!(stats1.distinct_states, stats2.distinct_states);
                assert_eq!(stats1.transitions, stats2.transitions);
            }
            (r1, r2) => panic!("Expected both to succeed, got {:?} and {:?}", r1, r2),
        }
    }

    #[test]
    fn test_simulation_max_trace_length() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Infinite spec (can always increment)
        let src = r#"
---- MODULE SimMaxLenTest ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x' = x + 1
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let sim_config = SimulationConfig {
            num_traces: 5,
            max_trace_length: 10,
            seed: Some(42),
            check_invariants: false,
            action_constraints: Vec::new(),
        };

        let mut checker = ModelChecker::new(&module, &config);
        let result = checker.simulate(&sim_config);

        match result {
            SimulationResult::Success(stats) => {
                assert_eq!(stats.traces_generated, 5);
                // All traces should hit the max length limit
                assert_eq!(stats.truncated_traces, 5);
                assert_eq!(stats.deadlocked_traces, 0);
                assert_eq!(stats.max_trace_length, 10);
            }
            other => panic!("Expected Success, got {:?}", other),
        }
    }

    #[test]
    fn test_simulation_no_invariant_checking() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Same spec as invariant violation test, but with invariant checking disabled
        let src = r#"
---- MODULE SimNoInvTest ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 5 /\ x' = x + 1
BadInvariant == x <= 2
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["BadInvariant".to_string()],
            ..Default::default()
        };

        let sim_config = SimulationConfig {
            num_traces: 10,
            max_trace_length: 100,
            seed: Some(42),
            check_invariants: false, // Disabled!
            action_constraints: Vec::new(),
        };

        let mut checker = ModelChecker::new(&module, &config);
        let result = checker.simulate(&sim_config);

        // Should succeed because we're not checking invariants
        match result {
            SimulationResult::Success(stats) => {
                assert_eq!(stats.traces_generated, 10);
            }
            other => panic!("Expected Success (no invariant check), got {:?}", other),
        }
    }

    // ============================
    // Action constraint tests
    // ============================

    #[test]
    fn test_action_constraint_filters_transitions() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Counter that can go up or down
        // Action constraint only allows increasing transitions
        let src = r#"
---- MODULE ActionConstraintTest ----
EXTENDS Integers
VARIABLE x

Init == x = 0

\* Can increase by 1 or decrease by 1
Next == x' \in {x - 1, x + 1} /\ x < 5

\* Without constraint: x can go -1, 0, 1, -1, 0, 1, ... (bounded by x < 5)
\* With OnlyIncrease: x can only go 0, 1, 2, 3, 4, 5 (all positive, then deadlock)

\* Only allow increasing transitions
OnlyIncrease == x' > x
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // With action constraint: should only see non-negative states
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            action_constraints: vec!["OnlyIncrease".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // With OnlyIncrease constraint, we should see states 0, 1, 2, 3, 4, 5
                // The spec bounds x < 5 so Next is disabled at x=5
                assert_eq!(stats.states_found, 6, "Should find exactly 6 states: 0-5");
            }
            other => panic!("Expected Success, got {:?}", other),
        }
    }

    #[test]
    fn test_action_constraint_primed_and_unprimed() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test that action constraints can reference both x (current) and x' (next)
        let src = r#"
---- MODULE ActionConstraintPrimeTest ----
EXTENDS Integers
VARIABLE x

Init == x = 0

\* Can change by any amount from -2 to +2, bounded by range
Next == x' \in {x - 2, x - 1, x, x + 1, x + 2} /\ x >= -2 /\ x <= 2

\* Action constraint: only allow changes of exactly 1 in either direction (or stutter)
\* Uses both x (current) and x' (next)
SmallStep == x' = x + 1 \/ x' = x - 1 \/ x' = x
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            action_constraints: vec!["SmallStep".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // With SmallStep constraint and bounded Next, we can reach -3 to +3
                // States: -3, -2, -1, 0, 1, 2, 3 = 7 states
                assert_eq!(
                    stats.states_found, 7,
                    "Should find exactly 7 states: -3 to +3"
                );
            }
            other => panic!("Expected Success, got {:?}", other),
        }
    }

    #[test]
    fn test_action_constraint_empty() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Verify that empty action constraints list means no filtering
        let src = r#"
---- MODULE NoActionConstraintTest ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == x < 3 /\ x' = x + 1
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            action_constraints: vec![], // Empty
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 4); // 0, 1, 2, 3
            }
            other => panic!("Expected Success, got {:?}", other),
        }
    }

    // ============================
    // Symmetry Reduction Tests
    // ============================

    #[test]
    fn test_symmetry_reduction_state_count() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // A spec with symmetric processes - without symmetry, each permutation
        // of process assignment would be a different state. With symmetry,
        // we identify symmetric states as equivalent.
        //
        // With 3 processes and one process being "active", without symmetry:
        // - Init: active = p1, active = p2, active = p3 = 3 initial states
        // With symmetry:
        // - Init: active = p1 (canonical representative) = 1 state
        let src = r#"
---- MODULE SymmetryTest ----
EXTENDS TLC
CONSTANT Procs
VARIABLE active

\* Each process can become active one at a time
Init == active \in Procs
Next == active' \in Procs /\ active' /= active

\* Symmetry: all permutations of Procs are equivalent
Sym == Permutations(Procs)
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config WITHOUT symmetry
        let mut config_no_sym = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        config_no_sym.constants.insert(
            "Procs".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec![
                "p1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
            ]),
        );

        // Config WITH symmetry
        let mut config_sym = config_no_sym.clone();
        config_sym.symmetry = Some("Sym".to_string());

        // Check WITHOUT symmetry
        let mut checker_no_sym = ModelChecker::new(&module, &config_no_sym);
        checker_no_sym.set_deadlock_check(false);
        let result_no_sym = checker_no_sym.check();

        let states_no_sym = match result_no_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success without symmetry, got {:?}", other),
        };

        // Check WITH symmetry
        let mut checker_sym = ModelChecker::new(&module, &config_sym);
        checker_sym.set_deadlock_check(false);
        let result_sym = checker_sym.check();

        let states_sym = match result_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success with symmetry, got {:?}", other),
        };

        // Verify symmetry reduces state count
        // Without symmetry: 3 states (active = p1, p2, or p3)
        // With symmetry: 1 state (all symmetric)
        assert_eq!(states_no_sym, 3, "Without symmetry should have 3 states");
        assert_eq!(states_sym, 1, "With symmetry should have 1 state");
        assert!(
            states_sym < states_no_sym,
            "Symmetry should reduce state count"
        );
    }

    #[test]
    fn test_symmetry_with_pairs() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // A spec where pairs of processes can be selected
        // Without symmetry: 3 choose 2 = 3 pairs, each pair assigned to one variable
        // With symmetry: only 1 canonical pair
        let src = r#"
---- MODULE SymmetryPairs ----
EXTENDS TLC, FiniteSets
CONSTANT Procs
VARIABLE selected

\* Select a subset of exactly 2 processes
Init == selected \in {S \in SUBSET Procs : Cardinality(S) = 2}
Next == UNCHANGED selected

\* Symmetry
Sym == Permutations(Procs)
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config WITHOUT symmetry
        let mut config_no_sym = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        config_no_sym.constants.insert(
            "Procs".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec![
                "p1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
            ]),
        );

        // Config WITH symmetry
        let mut config_sym = config_no_sym.clone();
        config_sym.symmetry = Some("Sym".to_string());

        // Check WITHOUT symmetry
        let mut checker_no_sym = ModelChecker::new(&module, &config_no_sym);
        checker_no_sym.set_deadlock_check(false);
        let result_no_sym = checker_no_sym.check();

        let states_no_sym = match result_no_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success without symmetry, got {:?}", other),
        };

        // Check WITH symmetry
        let mut checker_sym = ModelChecker::new(&module, &config_sym);
        checker_sym.set_deadlock_check(false);
        let result_sym = checker_sym.check();

        let states_sym = match result_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success with symmetry, got {:?}", other),
        };

        // Verify symmetry reduces state count
        // Without symmetry: 3 states ({p1,p2}, {p1,p3}, {p2,p3})
        // With symmetry: 1 state (all pairs are symmetric)
        assert_eq!(states_no_sym, 3, "Without symmetry should have 3 states");
        assert_eq!(states_sym, 1, "With symmetry should have 1 state");
    }

    #[test]
    fn test_symmetry_with_tuples() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test symmetry with model values inside tuples
        // A tuple <<p1, p2>> should be symmetric to <<p2, p1>> etc
        let src = r#"
---- MODULE SymmetryTuples ----
EXTENDS TLC
CONSTANT Procs
VARIABLE pair

\* Select an ordered pair of distinct processes
Init == pair \in {<<a, b>> : a \in Procs, b \in Procs \ {a}}
Next == UNCHANGED pair

\* Symmetry
Sym == Permutations(Procs)
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config WITHOUT symmetry
        let mut config_no_sym = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        config_no_sym.constants.insert(
            "Procs".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec![
                "p1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
            ]),
        );

        // Config WITH symmetry
        let mut config_sym = config_no_sym.clone();
        config_sym.symmetry = Some("Sym".to_string());

        // Check WITHOUT symmetry
        let mut checker_no_sym = ModelChecker::new(&module, &config_no_sym);
        checker_no_sym.set_deadlock_check(false);
        let result_no_sym = checker_no_sym.check();

        let states_no_sym = match result_no_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success without symmetry, got {:?}", other),
        };

        // Check WITH symmetry
        let mut checker_sym = ModelChecker::new(&module, &config_sym);
        checker_sym.set_deadlock_check(false);
        let result_sym = checker_sym.check();

        let states_sym = match result_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success with symmetry, got {:?}", other),
        };

        // Without symmetry: 3*2 = 6 ordered pairs (p1,p2), (p2,p1), (p1,p3), (p3,p1), (p2,p3), (p3,p2)
        // With symmetry: 1 canonical pair (all are symmetric)
        assert_eq!(
            states_no_sym, 6,
            "Without symmetry should have 6 ordered pairs"
        );
        assert_eq!(
            states_sym, 1,
            "With symmetry should have 1 canonical ordered pair"
        );
    }

    #[test]
    fn test_symmetry_with_records() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test symmetry with model values inside records
        let src = r#"
---- MODULE SymmetryRecords ----
EXTENDS TLC
CONSTANT Procs
VARIABLE msg

\* A message with sender and receiver fields
Init == msg \in [sender: Procs, receiver: Procs]
Next == UNCHANGED msg

\* Symmetry
Sym == Permutations(Procs)
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config WITHOUT symmetry
        let mut config_no_sym = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        config_no_sym.constants.insert(
            "Procs".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec![
                "p1".to_string(),
                "p2".to_string(),
            ]),
        );

        // Config WITH symmetry
        let mut config_sym = config_no_sym.clone();
        config_sym.symmetry = Some("Sym".to_string());

        // Check WITHOUT symmetry
        let mut checker_no_sym = ModelChecker::new(&module, &config_no_sym);
        checker_no_sym.set_deadlock_check(false);
        let result_no_sym = checker_no_sym.check();

        let states_no_sym = match result_no_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success without symmetry, got {:?}", other),
        };

        // Check WITH symmetry
        let mut checker_sym = ModelChecker::new(&module, &config_sym);
        checker_sym.set_deadlock_check(false);
        let result_sym = checker_sym.check();

        let states_sym = match result_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success with symmetry, got {:?}", other),
        };

        // Without symmetry: 2*2 = 4 records [sender:p1,receiver:p1], [s:p1,r:p2], [s:p2,r:p1], [s:p2,r:p2]
        // With symmetry: 2 canonical records (same/different sender-receiver)
        assert_eq!(
            states_no_sym, 4,
            "Without symmetry should have 4 record states"
        );
        assert_eq!(
            states_sym, 2,
            "With symmetry should have 2 canonical states (same/different)"
        );
    }

    #[test]
    fn test_symmetry_with_functions() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test symmetry with function values [Procs -> Values]
        let src = r#"
---- MODULE SymmetryFunctions ----
EXTENDS TLC
CONSTANT Procs
VARIABLE votes

\* Each process votes yes or no
Init == votes \in [Procs -> {"yes", "no"}]
Next == UNCHANGED votes

\* Symmetry
Sym == Permutations(Procs)
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config WITHOUT symmetry
        let mut config_no_sym = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        config_no_sym.constants.insert(
            "Procs".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
        );

        // Config WITH symmetry
        let mut config_sym = config_no_sym.clone();
        config_sym.symmetry = Some("Sym".to_string());

        // Check WITHOUT symmetry
        let mut checker_no_sym = ModelChecker::new(&module, &config_no_sym);
        checker_no_sym.set_deadlock_check(false);
        let result_no_sym = checker_no_sym.check();

        let states_no_sym = match result_no_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success without symmetry, got {:?}", other),
        };

        // Check WITH symmetry
        let mut checker_sym = ModelChecker::new(&module, &config_sym);
        checker_sym.set_deadlock_check(false);
        let result_sym = checker_sym.check();

        let states_sym = match result_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success with symmetry, got {:?}", other),
        };

        // Without symmetry: 2^2 = 4 functions
        // [p1->yes,p2->yes], [p1->yes,p2->no], [p1->no,p2->yes], [p1->no,p2->no]
        // With symmetry: 3 canonical functions
        // - both yes (or both no by relabeling)
        // - p1->yes, p2->no (symmetric to p1->no, p2->yes)
        // Actually: [yes,yes], [no,no], [yes,no] = 3 states
        assert_eq!(
            states_no_sym, 4,
            "Without symmetry should have 4 function states"
        );
        assert_eq!(
            states_sym, 3,
            "With symmetry should have 3 canonical states"
        );
    }

    #[test]
    fn test_symmetry_with_sequences() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test symmetry with sequences containing model values
        let src = r#"
---- MODULE SymmetrySequences ----
EXTENDS TLC, Sequences
CONSTANT Procs
VARIABLE queue

\* Queue contains a permutation of some processes
Init == queue \in {<<p>> : p \in Procs} \cup {<<p, q>> : p \in Procs, q \in Procs \ {p}}
Next == UNCHANGED queue

\* Symmetry
Sym == Permutations(Procs)
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config WITHOUT symmetry
        let mut config_no_sym = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        config_no_sym.constants.insert(
            "Procs".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec![
                "p1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
            ]),
        );

        // Config WITH symmetry
        let mut config_sym = config_no_sym.clone();
        config_sym.symmetry = Some("Sym".to_string());

        // Check WITHOUT symmetry
        let mut checker_no_sym = ModelChecker::new(&module, &config_no_sym);
        checker_no_sym.set_deadlock_check(false);
        let result_no_sym = checker_no_sym.check();

        let states_no_sym = match result_no_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success without symmetry, got {:?}", other),
        };

        // Check WITH symmetry
        let mut checker_sym = ModelChecker::new(&module, &config_sym);
        checker_sym.set_deadlock_check(false);
        let result_sym = checker_sym.check();

        let states_sym = match result_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success with symmetry, got {:?}", other),
        };

        // Without symmetry:
        // - Single element: <<p1>>, <<p2>>, <<p3>> = 3
        // - Two elements: <<p1,p2>>, <<p1,p3>>, <<p2,p1>>, <<p2,p3>>, <<p3,p1>>, <<p3,p2>> = 6
        // Total: 9 states
        // With symmetry:
        // - Single element: 1 canonical
        // - Two elements: 1 canonical (all ordered pairs symmetric)
        // Total: 2 canonical states
        assert_eq!(
            states_no_sym, 9,
            "Without symmetry should have 9 sequence states"
        );
        assert_eq!(
            states_sym, 2,
            "With symmetry should have 2 canonical states (length 1 and 2)"
        );
    }

    #[test]
    fn test_symmetry_large_permutation_group() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test with 4 elements to verify scaling: 4! = 24 permutations
        let src = r#"
---- MODULE SymmetryLarge ----
EXTENDS TLC
CONSTANT Procs
VARIABLE leader

\* Select a leader from processes
Init == leader \in Procs
Next == leader' \in Procs /\ leader' /= leader

\* Symmetry
Sym == Permutations(Procs)
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config WITHOUT symmetry
        let mut config_no_sym = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        config_no_sym.constants.insert(
            "Procs".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec![
                "p1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
                "p4".to_string(),
            ]),
        );

        // Config WITH symmetry
        let mut config_sym = config_no_sym.clone();
        config_sym.symmetry = Some("Sym".to_string());

        // Check WITHOUT symmetry
        let mut checker_no_sym = ModelChecker::new(&module, &config_no_sym);
        checker_no_sym.set_deadlock_check(false);
        let result_no_sym = checker_no_sym.check();

        let states_no_sym = match result_no_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success without symmetry, got {:?}", other),
        };

        // Check WITH symmetry
        let mut checker_sym = ModelChecker::new(&module, &config_sym);
        checker_sym.set_deadlock_check(false);
        let result_sym = checker_sym.check();

        let states_sym = match result_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success with symmetry, got {:?}", other),
        };

        // Without symmetry: 4 states (leader = p1, p2, p3, or p4)
        // With symmetry: 1 canonical state
        assert_eq!(
            states_no_sym, 4,
            "Without symmetry should have 4 states (4 processes)"
        );
        assert_eq!(states_sym, 1, "With symmetry should have 1 canonical state");
    }

    #[test]
    fn test_symmetry_preserves_invariant_violation() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Ensure symmetry reduction still catches invariant violations
        let src = r#"
---- MODULE SymmetryInvariant ----
EXTENDS TLC
CONSTANT Procs
VARIABLE active, count

Init == active \in Procs /\ count = 0
Next == /\ active' \in Procs
        /\ count' = count + 1

\* Invariant: count < 3 (will be violated)
Safety == count < 3

\* Symmetry
Sym == Permutations(Procs)
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config with invariant
        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safety".to_string()],
            ..Default::default()
        };
        config.constants.insert(
            "Procs".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
        );
        config.symmetry = Some("Sym".to_string());

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        let result = checker.check();

        // Should find invariant violation even with symmetry
        match result {
            CheckResult::InvariantViolation { .. } => {
                // Expected - symmetry doesn't hide violations
            }
            other => panic!(
                "Expected InvariantViolation with symmetry, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_symmetry_with_nested_structures() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test symmetry with deeply nested model values
        let src = r#"
---- MODULE SymmetryNested ----
EXTENDS TLC
CONSTANT Procs
VARIABLE state

\* Nested structure: set of records containing tuples of processes
Init == state \in {[leader |-> <<p1, p2>>] : p1 \in Procs, p2 \in Procs \ {p1}}
Next == UNCHANGED state

\* Symmetry
Sym == Permutations(Procs)
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config WITHOUT symmetry
        let mut config_no_sym = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        config_no_sym.constants.insert(
            "Procs".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec![
                "p1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
            ]),
        );

        // Config WITH symmetry
        let mut config_sym = config_no_sym.clone();
        config_sym.symmetry = Some("Sym".to_string());

        // Check WITHOUT symmetry
        let mut checker_no_sym = ModelChecker::new(&module, &config_no_sym);
        checker_no_sym.set_deadlock_check(false);
        let result_no_sym = checker_no_sym.check();

        let states_no_sym = match result_no_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success without symmetry, got {:?}", other),
        };

        // Check WITH symmetry
        let mut checker_sym = ModelChecker::new(&module, &config_sym);
        checker_sym.set_deadlock_check(false);
        let result_sym = checker_sym.check();

        let states_sym = match result_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success with symmetry, got {:?}", other),
        };

        // Without symmetry: 3*2 = 6 ordered pairs in nested structure
        // With symmetry: 1 canonical state
        assert_eq!(
            states_no_sym, 6,
            "Without symmetry should have 6 nested states"
        );
        assert_eq!(
            states_sym, 1,
            "With symmetry should have 1 canonical nested state"
        );
    }

    #[test]
    fn test_symmetry_multiple_variables() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Test symmetry with multiple state variables containing model values
        let src = r#"
---- MODULE SymmetryMultiVar ----
EXTENDS TLC
CONSTANT Procs
VARIABLE sender, receiver

Init == sender \in Procs /\ receiver \in Procs
Next == /\ sender' \in Procs
        /\ receiver' \in Procs
        /\ sender' /= sender

\* Symmetry
Sym == Permutations(Procs)
===="#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config WITHOUT symmetry
        let mut config_no_sym = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        config_no_sym.constants.insert(
            "Procs".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
        );

        // Config WITH symmetry
        let mut config_sym = config_no_sym.clone();
        config_sym.symmetry = Some("Sym".to_string());

        // Check WITHOUT symmetry
        let mut checker_no_sym = ModelChecker::new(&module, &config_no_sym);
        checker_no_sym.set_deadlock_check(false);
        let result_no_sym = checker_no_sym.check();

        let states_no_sym = match result_no_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success without symmetry, got {:?}", other),
        };

        // Check WITH symmetry
        let mut checker_sym = ModelChecker::new(&module, &config_sym);
        checker_sym.set_deadlock_check(false);
        let result_sym = checker_sym.check();

        let states_sym = match result_sym {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success with symmetry, got {:?}", other),
        };

        // Without symmetry: 2*2 = 4 states for (sender, receiver)
        // With symmetry: 2 canonical states (same or different)
        assert_eq!(
            states_no_sym, 4,
            "Without symmetry should have 4 multi-var states"
        );
        assert_eq!(
            states_sym, 2,
            "With symmetry should have 2 canonical states (same/different)"
        );
    }

    #[test]
    fn test_no_trace_mode_success() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == x' = (x + 1) % 3
TypeOK == x \in {0, 1, 2}
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        // Run with no-trace mode
        let mut checker = ModelChecker::new(&module, &config);
        checker.set_store_states(false);
        checker.set_deadlock_check(false);
        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 3, "Should find 3 states");
                // Verify internal state: seen_fps should be populated, seen should be empty
                assert!(
                    checker.seen.is_empty(),
                    "seen map should be empty in no-trace mode"
                );
                assert_eq!(checker.seen_fps.len(), 3, "seen_fps should have 3 entries");
            }
            other => panic!("Expected Success, got {:?}", other),
        }
    }

    #[test]
    fn test_no_trace_mode_violation_empty_trace() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
NeverTwo == x /= 2
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["NeverTwo".to_string()],
            ..Default::default()
        };

        // Run with no-trace mode (disables both full-state storage AND auto trace file)
        let mut checker = ModelChecker::new(&module, &config);
        checker.set_store_states(false);
        checker.set_auto_create_trace_file(false); // Required for truly empty traces
        checker.set_deadlock_check(false);
        let result = checker.check();

        match result {
            CheckResult::InvariantViolation {
                invariant, trace, ..
            } => {
                assert_eq!(invariant, "NeverTwo");
                // In no-trace mode (with auto trace file disabled), trace should be empty
                assert!(trace.is_empty(), "Trace should be empty in no-trace mode");
            }
            other => panic!("Expected InvariantViolation, got {:?}", other),
        }
    }

    #[test]
    fn test_trace_mode_vs_no_trace_mode_state_count() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == (x' = (x + 1) % 2) /\ (y' = (y + 1) % 2)
TypeOK == x \in {0, 1} /\ y \in {0, 1}
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        // Run with trace mode (default)
        let mut checker_trace = ModelChecker::new(&module, &config);
        checker_trace.set_deadlock_check(false);
        let result_trace = checker_trace.check();

        // Run with no-trace mode
        let mut checker_no_trace = ModelChecker::new(&module, &config);
        checker_no_trace.set_store_states(false);
        checker_no_trace.set_deadlock_check(false);
        let result_no_trace = checker_no_trace.check();

        // Both should find the same number of states
        let states_trace = match result_trace {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success (trace), got {:?}", other),
        };

        let states_no_trace = match result_no_trace {
            CheckResult::Success(stats) => stats.states_found,
            other => panic!("Expected Success (no-trace), got {:?}", other),
        };

        assert_eq!(
            states_trace, states_no_trace,
            "Both modes should find same state count"
        );
        // With the given Next relation, (0,0) -> (1,1) -> (0,0)... only 2 reachable states
        assert_eq!(states_trace, 2, "Should find 2 states: (0,0), (1,1)");
    }

    #[test]
    fn test_view_reduces_state_space() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Spec with two variables: x (main state) and counter (auxiliary)
        // Without VIEW: states distinguished by both x and counter
        // With VIEW: states only distinguished by x (counter ignored)
        let src = r#"
---- MODULE Test ----
VARIABLE x, counter
Init == x = 0 /\ counter = 0
Next == x' = (x + 1) % 3 /\ counter' = counter + 1
TypeOK == x \in {0, 1, 2}

\* VIEW expression: only consider x for fingerprinting
ViewX == x
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config WITHOUT VIEW
        let config_no_view = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        // Config WITH VIEW
        let mut config_with_view = config_no_view.clone();
        config_with_view.view = Some("ViewX".to_string());

        // Without VIEW: counter keeps incrementing, so we get many states
        // Actually, with counter growing unboundedly, we need a depth limit
        let mut checker_no_view = ModelChecker::new(&module, &config_no_view);
        checker_no_view.set_deadlock_check(false);
        checker_no_view.set_max_depth(5); // Limit depth to prevent infinite exploration
        let result_no_view = checker_no_view.check();

        let states_no_view = match result_no_view {
            CheckResult::Success(stats) => stats.states_found,
            CheckResult::LimitReached { stats, .. } => stats.states_found,
            other => panic!(
                "Expected Success or LimitReached without VIEW, got {:?}",
                other
            ),
        };

        // With VIEW: counter changes don't create new states (VIEW only looks at x)
        let mut checker_with_view = ModelChecker::new(&module, &config_with_view);
        checker_with_view.set_deadlock_check(false);
        checker_with_view.set_max_depth(5);
        let result_with_view = checker_with_view.check();

        let states_with_view = match result_with_view {
            CheckResult::Success(stats) => stats.states_found,
            CheckResult::LimitReached { stats, .. } => stats.states_found,
            other => panic!(
                "Expected Success or LimitReached with VIEW, got {:?}",
                other
            ),
        };

        // VIEW should reduce state count: only 3 distinct x values (0, 1, 2)
        assert_eq!(
            states_with_view, 3,
            "With VIEW, should only see 3 states (x values)"
        );

        // Without VIEW, we get more states (one per (x, counter) pair)
        // At depth 5, we get: initial + 5 successor states = 6 states
        assert!(
            states_no_view > states_with_view,
            "Without VIEW should have more states ({}) than with VIEW ({})",
            states_no_view,
            states_with_view
        );
    }

    #[test]
    fn test_checkpoint_save_during_bfs() {
        use tempfile::tempdir;
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Simple spec that generates several states
        let src = r#"
---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == x' = (x + 1) % 5
TypeOK == x \in 0..4
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        // Create checkpoint directory
        let checkpoint_dir = tempdir().unwrap();
        let checkpoint_path = checkpoint_dir.path().join("checkpoint");

        // Set up checker with short checkpoint interval
        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);
        // Use very short interval (0 seconds) to ensure checkpoint is saved
        checker.set_checkpoint(checkpoint_path.clone(), 0);
        checker.set_checkpoint_paths(Some("test.tla".to_string()), Some("test.cfg".to_string()));

        // Run model checking
        let result = checker.check();
        assert!(matches!(result, CheckResult::Success(_)));

        // Verify checkpoint was saved
        assert!(
            checkpoint_path.exists(),
            "Checkpoint directory should exist"
        );
        assert!(
            checkpoint_path.join("checkpoint.json").exists(),
            "Checkpoint metadata should exist"
        );
        assert!(
            checkpoint_path.join("fingerprints.bin").exists(),
            "Fingerprints file should exist"
        );
        assert!(
            checkpoint_path.join("frontier.json").exists(),
            "Frontier file should exist"
        );
    }

    #[test]
    fn test_checkpoint_resume_continues_exploration() {
        use tempfile::tempdir;
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Spec with branching but NO cycles, so a resume must correctly account for
        // already-seen states (it can't rediscover earlier states via transitions).
        let src = r#"
---- MODULE Test ----
VARIABLE x, y
Init == x = 0 /\ y = 0
Next ==
    \/ (x < 2 /\ x' = x + 1 /\ y' = y)
    \/ (y < 2 /\ x' = x /\ y' = y + 1)
TypeOK == x \in 0..2 /\ y \in 0..2
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };

        // First run without checkpoint to get baseline state count
        let expected_states = {
            let mut checker = ModelChecker::new(&module, &config);
            checker.set_deadlock_check(false);
            let result = checker.check();
            match result {
                CheckResult::Success(stats) => stats.states_found,
                other => panic!("Expected Success, got {:?}", other),
            }
        };

        assert_eq!(expected_states, 9, "Expected 3x3 grid to have 9 states");

        // Now test checkpoint/resume by stopping early with a states limit
        let checkpoint_dir = tempdir().unwrap();
        let checkpoint_path = checkpoint_dir.path().join("checkpoint");

        // First, run and let it save checkpoint, but stop early.
        {
            let mut checker = ModelChecker::new(&module, &config);
            checker.set_deadlock_check(false);
            checker.set_max_states(3);
            checker.set_checkpoint(checkpoint_path.clone(), 0);

            let result = checker.check();
            assert!(matches!(
                result,
                CheckResult::LimitReached {
                    limit_type: LimitType::States,
                    ..
                }
            ));
        }

        // Verify checkpoint was saved and has a non-empty frontier to resume from
        assert!(checkpoint_path.exists());
        {
            use crate::checkpoint::Checkpoint;
            let checkpoint = Checkpoint::load(&checkpoint_path).expect("checkpoint load");
            assert!(
                !checkpoint.frontier.is_empty(),
                "Expected checkpoint to contain pending frontier states"
            );
            assert!(
                checkpoint.metadata.stats.states_found >= 3,
                "Expected checkpoint to record progress"
            );
        }

        // Now test that check_with_resume continues exploration to completion.
        {
            let mut checker = ModelChecker::new(&module, &config);
            checker.set_deadlock_check(false);

            let result = checker
                .check_with_resume(&checkpoint_path)
                .expect("Resume should succeed");

            match result {
                CheckResult::Success(stats) => {
                    assert_eq!(
                        stats.states_found, expected_states,
                        "Resumed run should reach full state count"
                    );
                    assert_eq!(
                        checker.seen_fps.len(),
                        expected_states,
                        "seen_fps should reflect total explored states"
                    );
                }
                other => panic!("Expected Success after resume, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_terminal_states_not_reported_as_deadlock() {
        use crate::config::TerminalSpec;
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Simple spec that terminates at state = "SAT" or state = "UNSAT"
        // Without TERMINAL directive, this would be a deadlock
        let src = r#"
---- MODULE SATSolver ----
VARIABLE state
Init == state = "searching"
Next ==
    \/ (state = "searching" /\ state' = "SAT")
    \/ (state = "searching" /\ state' = "UNSAT")
TypeOK == state \in {"searching", "SAT", "UNSAT"}
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config WITHOUT terminal - should report deadlock
        let config_no_terminal = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            check_deadlock: true,
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config_no_terminal);
        let result = checker.check();

        // Without TERMINAL, this should deadlock at SAT or UNSAT
        assert!(
            matches!(result, CheckResult::Deadlock { .. }),
            "Without TERMINAL, should report deadlock. Got: {:?}",
            result
        );

        // Config WITH terminal predicates - should NOT report deadlock
        let config_with_terminal = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            check_deadlock: true,
            terminal: Some(TerminalSpec::Predicates(vec![
                ("state".to_string(), "\"SAT\"".to_string()),
                ("state".to_string(), "\"UNSAT\"".to_string()),
            ])),
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config_with_terminal);
        let result = checker.check();

        // With TERMINAL, should succeed (SAT/UNSAT are terminal states)
        match result {
            CheckResult::Success(stats) => {
                // Should have 3 states: searching, SAT, UNSAT
                assert_eq!(
                    stats.states_found, 3,
                    "Should find 3 states: searching, SAT, UNSAT"
                );
            }
            other => panic!(
                "With TERMINAL predicates, expected Success but got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_terminal_operator() {
        use crate::config::TerminalSpec;
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Spec with IsTerminal operator
        let src = r#"
---- MODULE TerminalOp ----
VARIABLE x
Init == x = 0
Next == x < 3 /\ x' = x + 1
TypeOK == x \in 0..3
IsTerminal == x = 3
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Config with terminal operator
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            check_deadlock: true,
            terminal: Some(TerminalSpec::Operator("IsTerminal".to_string())),
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        let result = checker.check();

        // With TERMINAL IsTerminal, x=3 should not be a deadlock
        match result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 4, "Should find 4 states: x=0,1,2,3");
            }
            other => panic!("Expected Success with TERMINAL operator, got {:?}", other),
        }
    }

    /// Test SpanTree-like pattern: nested EXISTS with guard at intermediate level
    /// and two variables updated by assignments depending on different EXISTS levels.
    ///
    /// This pattern caused TLA2 to miss states because the `mom` assignment
    /// depends on `m` from the outer EXISTS, while `dist` depends on `d` from inner EXISTS.
    #[test]
    fn test_spantree_nested_exists_two_variables() {
        use crate::config::ConstantValue;
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Simplified SpanTree pattern with 3 nodes
        // TLC finds 22 states for this configuration
        let src = r#"
---- MODULE SpanTreeMinimal ----
EXTENDS Integers

CONSTANTS Nodes, Edges, MaxCardinality, Root

Nbrs(n) == {m \in Nodes : {m, n} \in Edges}

VARIABLES mom, dist

Init == /\ mom = [n \in Nodes |-> n]
        /\ dist = [n \in Nodes |-> IF n = Root THEN 0 ELSE MaxCardinality]

Next == \E n \in Nodes :
          \E m \in Nbrs(n) :
             /\ dist[m] < 1 + dist[n]
             /\ \E d \in (dist[m]+1) .. (dist[n] - 1) :
                    /\ dist' = [dist EXCEPT ![n] = d]
                    /\ mom'  = [mom  EXCEPT ![n] = m]

Spec == Init /\ [][Next]_<<mom, dist>>
====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.unwrap();

        // Set up constants: Nodes = {1,2,3}, Edges = {{1,2},{1,3},{2,3}}, MaxCardinality = 4, Root = 1
        let mut constants = std::collections::HashMap::new();
        constants.insert(
            "Nodes".to_string(),
            ConstantValue::Value("{1, 2, 3}".to_string()),
        );
        constants.insert(
            "Edges".to_string(),
            ConstantValue::Value("{{1, 2}, {1, 3}, {2, 3}}".to_string()),
        );
        constants.insert(
            "MaxCardinality".to_string(),
            ConstantValue::Value("4".to_string()),
        );
        constants.insert("Root".to_string(), ConstantValue::Value("1".to_string()));

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constants,
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // TLC finds 22 distinct states for this configuration
                // If TLA2 finds fewer, there's a bug in nested EXISTS enumeration
                assert_eq!(
                    stats.states_found, 22,
                    "SpanTree should find 22 states (TLC baseline). Found {}",
                    stats.states_found
                );
            }
            other => panic!("Expected success, got: {:?}", other),
        }
    }

    /// Regression test for bug #458: LET-bound variables not substituted in symbolic assignments
    ///
    /// When state variable updates occur inside LET expressions, the LET-bound variables
    /// must be substituted into the expression before it's stored as a SymbolicAssignment::Expr.
    /// Otherwise, when the expression is later evaluated, the LET-bound variables won't be in scope.
    ///
    /// This was causing "Undefined variable: s" errors in specs like MultiPaxos where
    /// PlusCal `with` clauses translated to LET expressions containing state updates.
    #[test]
    fn test_let_bound_var_substitution_in_state_update() {
        use tla_core::{lower, parse_to_syntax_tree, FileId};

        // Minimal repro: LET binds 's' and 'c', which are used in the msgs' update
        let src = r#"
---- MODULE LetBoundVarSubst ----
VARIABLE msgs

Init == msgs = {}

\* When EXISTS binds 's' and 'c', they're in a LET scope during extraction
\* The state update uses these variables
Next ==
    \E s \in {"server1", "server2"} :
    \E c \in {"cmd1", "cmd2"} :
    LET newMsg == [src |-> s, cmd |-> c]
    IN msgs' = msgs \cup {newMsg}

====
"#;
        let tree = parse_to_syntax_tree(src);
        let lower_result = lower(FileId(0), &tree);
        let module = lower_result.module.expect("Module should parse");

        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_deadlock_check(false);

        let result = checker.check();

        match result {
            CheckResult::Success(stats) => {
                // 1 initial state + 4 states from 2x2 combinations = 5 states total
                // (Actually more because msgs accumulates: {}, {m1}, {m1,m2}, etc.)
                assert!(
                    stats.states_found >= 5,
                    "Should find at least 5 states, found {}",
                    stats.states_found
                );
            }
            CheckResult::Error { error, .. } => {
                panic!(
                    "LET-bound variables not properly substituted! Error: {:?}",
                    error
                );
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }

    // ============================================================================
    // SNAPSHOT TESTS - CheckError message format stability
    // These tests ensure error messages don't change unexpectedly.
    // ============================================================================

    #[test]
    fn snapshot_check_error_missing_init() {
        use insta::assert_snapshot;
        assert_snapshot!(CheckError::MissingInit.to_string());
    }

    #[test]
    fn snapshot_check_error_missing_next() {
        use insta::assert_snapshot;
        assert_snapshot!(CheckError::MissingNext.to_string());
    }

    #[test]
    fn snapshot_check_error_missing_invariant() {
        use insta::assert_snapshot;
        let err = CheckError::MissingInvariant("SafetyProperty".to_string());
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_check_error_missing_property() {
        use insta::assert_snapshot;
        let err = CheckError::MissingProperty("LivenessProperty".to_string());
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_check_error_init_not_boolean() {
        use insta::assert_snapshot;
        assert_snapshot!(CheckError::InitNotBoolean.to_string());
    }

    #[test]
    fn snapshot_check_error_next_not_boolean() {
        use insta::assert_snapshot;
        assert_snapshot!(CheckError::NextNotBoolean.to_string());
    }

    #[test]
    fn snapshot_check_error_invariant_not_boolean() {
        use insta::assert_snapshot;
        let err = CheckError::InvariantNotBoolean("TypeOK".to_string());
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_check_error_no_variables() {
        use insta::assert_snapshot;
        assert_snapshot!(CheckError::NoVariables.to_string());
    }

    #[test]
    fn snapshot_check_error_init_cannot_enumerate() {
        use insta::assert_snapshot;
        let err =
            CheckError::InitCannotEnumerate("Variable 'x' has infinite domain (Nat)".to_string());
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_check_error_liveness() {
        use insta::assert_snapshot;
        let err = CheckError::LivenessError("Temporal property Liveness violated".to_string());
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_check_error_fingerprint_overflow() {
        use insta::assert_snapshot;
        let err = CheckError::FingerprintStorageOverflow { dropped: 1000 };
        assert_snapshot!(err.to_string());
    }

    #[test]
    fn snapshot_check_error_assume_false() {
        use insta::assert_snapshot;
        let err = CheckError::AssumeFalse {
            location: "line 5, col 1 to line 5, col 20 of module Test".to_string(),
        };
        assert_snapshot!(err.to_string());
    }
}
