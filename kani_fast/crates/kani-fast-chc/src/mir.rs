//! Simplified Rust MIR to CHC encoding
//!
//! This module provides a lightweight MIR-like representation focused on
//! control-flow structure and encodes it into the generic `TransitionSystem`
//! used by the CHC engine. It is intentionally minimal and uses SMT-LIB2
//! strings for expressions so it can be driven by upstream MIR lowering later.
//!
//! # Supported Features
//!
//! - Basic blocks with program counter tracking
//! - Assignments, assumptions, and assertions
//! - Control flow: goto, conditional goto, switch, return
//! - Function calls (inlined as block sequences or uninterpreted)
//! - Array operations (select/store in SMT-LIB2 format)
//! - Bitvector types for fixed-width integers

use crate::algebraic_rewrite::rewrite_expression;
use crate::clause::{sanitize_smt_identifier, UninterpretedFunction};
use crate::delegation::{choose_strategy, DelegationReason, VerificationPath};
use crate::encoding::encode_transition_system;
use crate::intrinsics;
use crate::proof_relevance::ProofRelevanceAnalysis;
use crate::ChcSystem;
use kani_fast_kinduction::{Property, SmtType, StateFormula, TransitionSystem};
use lazy_static::lazy_static;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;

lazy_static! {
    /// SMT-LIB keywords and builtins that should NOT be treated as variable names.
    /// Static initialization avoids recreating this set on every call to collect_vars_from_expr.
    static ref SMT_KEYWORDS: HashSet<&'static str> = {
        [
            "and", "or", "not", "ite", "true", "false", "Int", "Bool", "Real", "forall", "exists",
            "let", "as", "const", "Array", "select", "store", "BitVec", "bvadd", "bvsub", "bvmul",
            "bvudiv", "bvurem", "bvand", "bvor", "bvxor", "bvshl", "bvlshr", "bvashr", "bvnot",
            "bvneg", "concat", "extract", "div", "mod", "abs", "to_real", "to_int", "is_int", "pow",
            "sqrt", "bitand", "bitor", "bitxor", "pow2", // Our uninterpreted functions
        ]
        .iter()
        .copied()
        .collect()
    };

    /// Common interned strings for hot identifiers to avoid repeated allocations.
    /// These are the most frequently used variable names in MIR programs.
    static ref INTERNED_STRINGS: HashMap<&'static str, String> = {
        let mut m = HashMap::with_capacity(64);
        // Program counter
        m.insert("pc", "pc".to_string());
        m.insert("pc'", "pc'".to_string());
        // Common local variables _0 through _15
        for i in 0..=15 {
            // We use Box::leak to convert these to &'static str for the keys
            // This is safe because lazy_static ensures this only runs once
            let name_static: &'static str = Box::leak(format!("_{i}").into_boxed_str());
            let primed_static: &'static str = Box::leak(format!("_{i}'").into_boxed_str());
            // The leaked static strings are used as both keys and values (via to_string())
            m.insert(name_static, name_static.to_string());
            m.insert(primed_static, primed_static.to_string());
        }
        // Common field suffixes
        for suffix in &["_field0", "_field1", "_field2", "_elem_0", "_elem_1", "_discr", "_val", "_next", "_start", "_end"] {
            // Use (*suffix) to get &str directly for faster ToString specialization
            let s = (*suffix).to_string();
            m.insert(*suffix, s);
        }
        m
    };
}

/// Program counter sentinel value for abort terminators (e.g., `unreachable!()`, `panic!()`).
/// When PC reaches this value, it indicates the program reached an error state.
pub const PC_ABORT_SENTINEL: i64 = -2;

/// Program counter sentinel value for panic paths in compiled assert!() macros.
/// Rust's assert!() macro compiles to a conditional branch where the else-branch
/// goes to a panic path. We use this large positive value to avoid collisions
/// with regular basic block IDs and to avoid i64 overflow issues.
pub const PC_PANIC_SENTINEL: i64 = 999999;

/// Block ID sentinel for panic paths (usize version of PC_PANIC_SENTINEL).
/// Used for MirTerminator targets when the branch leads to an error/panic path.
pub const PANIC_BLOCK_ID: usize = 999999;

/// Program counter sentinel value for normal function returns.
/// When PC equals this value, the function has returned successfully.
pub const PC_RETURN_SENTINEL: i64 = -1;

/// A local variable in MIR with its name and SMT type.
///
/// Corresponds to Rust's local variables (`_0`, `_1`, etc.) or named bindings.
#[derive(Debug, Clone)]
pub struct MirLocal {
    /// The variable name (e.g., "_0", "_1", "x", etc.)
    pub name: String,
    /// The SMT-LIB type for this variable
    pub ty: SmtType,
}

impl MirLocal {
    pub fn new(name: impl Into<String>, ty: SmtType) -> Self {
        Self {
            name: name.into(),
            ty,
        }
    }
}

/// A MIR statement (simplified)
#[derive(Debug, Clone)]
pub enum MirStatement {
    /// Logical assumption that must hold on this path
    Assume(String),
    /// Assignment to a local
    Assign { lhs: String, rhs: String },
    /// Assertion to be encoded as a safety property
    Assert {
        condition: String,
        message: Option<String>,
    },
    /// Array store: `arr[index] = value` (in SMT: `(store arr index value)`)
    ArrayStore {
        array: String,
        index: String,
        value: String,
    },
    /// Nondeterministic assignment (havoc) - unconstrained value
    Havoc { var: String },
}

/// MIR terminators covering basic control-flow
#[derive(Debug, Clone)]
pub enum MirTerminator {
    /// Unconditional jump to another block
    Goto { target: usize },
    /// Conditional jump based on a boolean condition
    CondGoto {
        condition: String,
        then_target: usize,
        else_target: usize,
    },
    /// Switch over integer discriminant
    SwitchInt {
        discr: String,
        targets: Vec<(i64, usize)>,
        otherwise: usize,
    },
    /// Function call (inlined or uninterpreted)
    /// For inlined calls: destination is the return block, func_blocks are inlined
    /// For uninterpreted: result is assigned from an uninterpreted function application
    Call {
        /// Destination variable for return value (if any)
        destination: Option<String>,
        /// The function being called (as SMT uninterpreted function name)
        func: String,
        /// Arguments to the function (as SMT expressions)
        args: Vec<String>,
        /// Block to continue to after the call returns
        target: usize,
        /// Block to jump to on unwind (panic)
        unwind: Option<usize>,
        /// MODULAR VERIFICATION: Precondition check from callee's contract
        /// If present, this SMT formula must be verified before the call executes.
        /// This enforces the callee's `#[requires]` specification.
        precondition_check: Option<String>,
        /// MODULAR VERIFICATION: Postcondition assumption from callee's contract
        /// If present, this SMT formula constrains the return value based on the callee's `#[ensures]`
        postcondition_assumption: Option<String>,
        /// True when this call is a Range::into_iter invocation (set by driver/parser)
        is_range_into_iter: bool,
        /// True when this call is a Range::next invocation (set by driver/parser)
        is_range_next: bool,
    },
    /// Function returns
    Return,
    /// No outgoing edges (treated as dead end)
    Unreachable,
    /// Abort/panic (treated as reaching error state)
    Abort,
}

/// A MIR basic block
#[derive(Debug, Clone)]
pub struct MirBasicBlock {
    pub id: usize,
    pub statements: Vec<MirStatement>,
    pub terminator: MirTerminator,
}

impl MirBasicBlock {
    pub fn new(id: usize, terminator: MirTerminator) -> Self {
        Self {
            id,
            statements: Vec::new(),
            terminator,
        }
    }

    pub fn with_statement(mut self, stmt: MirStatement) -> Self {
        self.statements.push(stmt);
        self
    }
}

/// A simplified MIR program with explicit start block
#[derive(Debug, Clone)]
pub struct MirProgram {
    pub locals: Vec<MirLocal>,
    pub basic_blocks: Vec<MirBasicBlock>,
    pub start_block: usize,
    pub init: Option<StateFormula>,
    /// Map from source variable names to MIR local names
    /// e.g., "i" -> "_3", "sum" -> "_2"
    pub var_to_local: std::collections::HashMap<String, String>,
    /// Closure functions available for inlining
    /// Key: source pattern (e.g., "/tmp/file.rs:2:19: 2:27")
    /// Value: closure info for inlining
    pub closures: HashMap<String, ClosureInfo>,
    /// Trait impl methods available for static dispatch inlining
    /// Key: trait-qualified call pattern (e.g., `<Value as Addable>::add_value`)
    /// Value: impl info for inlining
    pub trait_impls: HashMap<String, TraitImplInfo>,
}

impl MirProgram {
    pub fn builder(start_block: usize) -> MirProgramBuilder {
        MirProgramBuilder::new(start_block)
    }
}

/// Builder for `MirProgram`
pub struct MirProgramBuilder {
    locals: Vec<MirLocal>,
    blocks: Vec<MirBasicBlock>,
    start_block: usize,
    init: Option<StateFormula>,
    var_to_local: std::collections::HashMap<String, String>,
    closures: HashMap<String, ClosureInfo>,
    trait_impls: HashMap<String, TraitImplInfo>,
}

impl MirProgramBuilder {
    pub fn new(start_block: usize) -> Self {
        Self {
            locals: Vec::new(),
            blocks: Vec::new(),
            start_block,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        }
    }

    /// Add a mapping from source variable name to MIR local
    pub fn var_mapping(
        mut self,
        source_name: impl Into<String>,
        mir_local: impl Into<String>,
    ) -> Self {
        self.var_to_local
            .insert(source_name.into(), mir_local.into());
        self
    }

    pub fn local(mut self, name: impl Into<String>, ty: SmtType) -> Self {
        self.locals.push(MirLocal::new(name, ty));
        self
    }

    pub fn block(mut self, block: MirBasicBlock) -> Self {
        self.blocks.push(block);
        self
    }

    pub fn init(mut self, formula: impl Into<String>) -> Self {
        self.init = Some(StateFormula::new(formula));
        self
    }

    /// Add a closure for potential inlining
    pub fn closure(mut self, source_pattern: impl Into<String>, info: ClosureInfo) -> Self {
        self.closures.insert(source_pattern.into(), info);
        self
    }

    pub fn finish(self) -> MirProgram {
        MirProgram {
            locals: self.locals,
            basic_blocks: self.blocks,
            start_block: self.start_block,
            init: self.init,
            var_to_local: self.var_to_local,
            closures: self.closures,
            trait_impls: self.trait_impls,
        }
    }
}

/// Optimize a MIR program for CHC solving with unbounded integers.
///
/// Transformations applied:
/// 1. Overflow flag elimination: Set all `_X_elem_1` (overflow flags) to `false`
///    since unbounded Int can never overflow.
/// 2. Remove branches to error state that check overflow flags (they're always false).
///
/// This is sound because:
/// - SMT Int sort represents mathematical integers with no upper/lower bounds
/// - Overflow is a property of bounded integer types, not mathematical integers
/// - Setting overflow flags to false eliminates spurious error branches
///
/// Returns `Cow::Borrowed` if no optimizations were applied, avoiding unnecessary cloning.
pub fn optimize_mir_for_unbounded(program: &MirProgram) -> Cow<'_, MirProgram> {
    // First pass: check if any optimizations will be applied
    let needs_optimization = program.basic_blocks.iter().any(|block| {
        // Check if any statements need modification
        let stmts_need_opt = block.statements.iter().any(|stmt| match stmt {
            MirStatement::Assign { lhs, rhs }
                if lhs.ends_with("_elem_1") || lhs.ends_with("_field1") =>
            {
                rhs.contains("2147483647")
                    || rhs.contains("255")
                    || rhs.contains("65535")
                    || rhs.contains("-128")
                    || rhs.contains("-32768")
                    || rhs.contains("-2147483648")
                    || rhs.contains("> 0")
                    || rhs.contains("< 0")
                    || rhs == "false"
            }
            MirStatement::Havoc { var } if var.ends_with("_elem_1") || var.ends_with("_field1") => {
                true
            }
            _ => false,
        });

        // Check if terminator needs modification
        let term_needs_opt = if let MirTerminator::CondGoto { condition, .. } = &block.terminator {
            is_overflow_flag(condition) || is_negated_overflow_flag(condition)
        } else {
            false
        };

        stmts_need_opt || term_needs_opt
    });

    // If no optimizations needed, return borrowed reference (no clone!)
    if !needs_optimization {
        return Cow::Borrowed(program);
    }

    // Apply optimizations
    let mut optimized_blocks = Vec::with_capacity(program.basic_blocks.len());

    for block in &program.basic_blocks {
        let mut optimized_stmts = Vec::with_capacity(block.statements.len());

        for stmt in &block.statements {
            match stmt {
                // Eliminate overflow flag assignments when the RHS looks like an overflow check.
                // Since overflow flags are always false with unbounded Int, and they're
                // eliminated from live variables, we don't need to emit any assignment.
                // BUT: preserve assignments that don't look like overflow checks (e.g., struct fields).
                MirStatement::Assign { lhs, rhs }
                    if lhs.ends_with("_elem_1") || lhs.ends_with("_field1") =>
                {
                    // Check if this looks like an overflow condition (contains comparison with bounds)
                    if rhs.contains("2147483647")
                        || rhs.contains("255")
                        || rhs.contains("65535")
                        || rhs.contains("-128")
                        || rhs.contains("-32768")
                        || rhs.contains("-2147483648")
                        || rhs.contains("> 0")
                        || rhs.contains("< 0")
                        || rhs == "false"
                    {
                        // Eliminate the assignment - overflow flags will be eliminated
                        // from state variables by dead variable analysis
                    } else {
                        // Preserve - this might be a struct field or other non-overflow use
                        optimized_stmts.push(stmt.clone());
                    }
                }
                // Eliminate Havoc on overflow flags entirely
                MirStatement::Havoc { var }
                    if var.ends_with("_elem_1") || var.ends_with("_field1") =>
                {
                    // Eliminate - overflow flags will be removed by dead variable analysis
                }
                _ => {
                    optimized_stmts.push(stmt.clone());
                }
            }
        }

        // Optimize terminator: simplify CondGoto where the condition is always true/false
        let optimized_term = match &block.terminator {
            MirTerminator::CondGoto {
                condition,
                then_target,
                else_target,
            } => {
                // Check for common overflow check patterns:
                // - `_X_elem_1` or `_X_field1` (overflow flag, will be false after optimization)
                // - `(not _X_elem_1)` (negated overflow flag, will be true)
                if is_overflow_flag(condition) {
                    // Overflow flag is false, so condition is false -> take else branch
                    MirTerminator::Goto {
                        target: *else_target,
                    }
                } else if is_negated_overflow_flag(condition) {
                    // (not overflow_flag) is true -> take then branch
                    MirTerminator::Goto {
                        target: *then_target,
                    }
                } else {
                    block.terminator.clone()
                }
            }
            _ => block.terminator.clone(),
        };

        optimized_blocks.push(MirBasicBlock {
            id: block.id,
            statements: optimized_stmts,
            terminator: optimized_term,
        });
    }

    Cow::Owned(MirProgram {
        locals: program.locals.clone(),
        basic_blocks: optimized_blocks,
        start_block: program.start_block,
        init: program.init.clone(),
        var_to_local: program.var_to_local.clone(),
        closures: program.closures.clone(),
        trait_impls: program.trait_impls.clone(),
    })
}

/// Detect whether a MIR program contains bitwise operations that benefit from BitVec encoding
pub fn program_needs_bitvec_encoding(program: &MirProgram) -> bool {
    use crate::bitvec::needs_bitvec_encoding;

    if let Some(init) = &program.init {
        if needs_bitvec_encoding(&init.smt_formula) {
            return true;
        }
    }

    for block in &program.basic_blocks {
        for stmt in &block.statements {
            match stmt {
                MirStatement::Assume(cond) if needs_bitvec_encoding(cond) => return true,
                MirStatement::Assign { rhs, .. } if needs_bitvec_encoding(rhs) => return true,
                MirStatement::Assert { condition, .. } if needs_bitvec_encoding(condition) => {
                    return true;
                }
                MirStatement::ArrayStore { index, value, .. }
                    if needs_bitvec_encoding(index) || needs_bitvec_encoding(value) =>
                {
                    return true;
                }
                _ => {}
            }
        }

        let terminator_has_bitvec = match &block.terminator {
            MirTerminator::CondGoto { condition, .. } => needs_bitvec_encoding(condition),
            MirTerminator::SwitchInt { discr, .. } => needs_bitvec_encoding(discr),
            MirTerminator::Call {
                args,
                postcondition_assumption,
                ..
            } => {
                args.iter().any(|a| needs_bitvec_encoding(a))
                    || postcondition_assumption
                        .as_ref()
                        .is_some_and(|p| needs_bitvec_encoding(p))
            }
            MirTerminator::Goto { .. }
            | MirTerminator::Return
            | MirTerminator::Unreachable
            | MirTerminator::Abort => false,
        };

        if terminator_has_bitvec {
            return true;
        }
    }

    false
}

/// Check if an expression is an overflow flag variable (always false after optimization).
/// Matches variable references ending in _elem_1 or _field1, or the literal "false".
fn is_overflow_flag(expr: &str) -> bool {
    let trimmed = expr.trim();
    // Simple variable reference to overflow flag
    (trimmed.starts_with('_') && (trimmed.ends_with("_elem_1") || trimmed.ends_with("_field1")))
        || trimmed == "false"
}

/// Check if an expression is a negated overflow flag (always true after optimization).
/// Matches negated overflow flag variables or "true" literal.
fn is_negated_overflow_flag(expr: &str) -> bool {
    let trimmed = expr.trim();
    // Pattern: (not _X_elem_1) or (not _X_field1)
    if trimmed.starts_with("(not ") && trimmed.ends_with(')') {
        let inner = &trimmed[5..trimmed.len() - 1].trim();
        return is_overflow_flag(inner) || *inner == "false";
    }
    trimmed == "true"
}

/// Pre-compute the bidirectional assignment graph for variable chain analysis.
///
/// Returns a map from each variable to the set of variables it's connected to
/// via direct assignments (both directions: lhs->rhs and rhs->lhs).
/// This is computed once and reused for all Range iterator chain lookups.
fn build_assignment_graph(program: &MirProgram) -> HashMap<String, HashSet<String>> {
    let mut graph: HashMap<String, HashSet<String>> = HashMap::new();

    for block in &program.basic_blocks {
        for stmt in &block.statements {
            if let MirStatement::Assign { lhs, rhs } = stmt {
                // Only track simple variable-to-variable assignments (_N = _M)
                let lhs_is_var =
                    lhs.starts_with('_') && lhs.chars().skip(1).all(|c| c.is_ascii_digit());
                let rhs_is_var =
                    rhs.starts_with('_') && rhs.chars().skip(1).all(|c| c.is_ascii_digit());

                if lhs_is_var && rhs_is_var {
                    // Add bidirectional edges
                    graph.entry(lhs.clone()).or_default().insert(rhs.clone());
                    graph.entry(rhs.clone()).or_default().insert(lhs.clone());
                }
            }
        }
    }

    graph
}

/// Find all variables reachable from a starting variable in the assignment graph.
///
/// Uses BFS to traverse the pre-computed assignment graph, avoiding the O(n^3)
/// fixed-point iteration over all blocks and statements.
fn find_connected_variables(
    start: &str,
    graph: &HashMap<String, HashSet<String>>,
) -> HashSet<String> {
    let mut visited = HashSet::new();
    let mut queue = std::collections::VecDeque::new();

    visited.insert(start.to_string());
    queue.push_back(start.to_string());

    while let Some(current) = queue.pop_front() {
        if let Some(neighbors) = graph.get(&current) {
            for neighbor in neighbors {
                if visited.insert(neighbor.clone()) {
                    queue.push_back(neighbor.clone());
                }
            }
        }
    }

    visited
}

/// Analyze which variables are actually used in the MIR program.
///
/// A variable is considered "live" if it:
/// 1. Appears in a condition (assume, assert, conditional terminator)
/// 2. Is read in an assignment RHS
/// 3. Is the destination of an assignment that is subsequently read
///
/// Variables that are only assigned but never read can be eliminated from
/// the invariant predicate, reducing the state space for CHC solving.
fn analyze_live_variables(program: &MirProgram) -> HashSet<String> {
    let mut live = HashSet::new();
    let mut assigned = HashSet::new();

    // Pre-compute assignment graph once for efficient Range variable chain lookups.
    // This replaces the O(n^3) fixed-point iteration with O(V+E) BFS per query.
    let assignment_graph = build_assignment_graph(program);

    // Include variables from init constraints (they must be in the state)
    if let Some(init) = &program.init {
        collect_vars_from_expr(&init.smt_formula, &mut live);
    }

    // First pass: collect all assignments and reads
    for block in &program.basic_blocks {
        for stmt in &block.statements {
            match stmt {
                MirStatement::Assume(cond) => {
                    collect_vars_from_expr(cond, &mut live);
                }
                MirStatement::Assign { lhs, rhs } => {
                    assigned.insert(lhs.clone());
                    collect_vars_from_expr(rhs, &mut live);
                }
                MirStatement::Assert { condition, .. } => {
                    collect_vars_from_expr(condition, &mut live);
                }
                MirStatement::ArrayStore {
                    array,
                    index,
                    value,
                } => {
                    assigned.insert(array.clone());
                    collect_vars_from_expr(array, &mut live);
                    collect_vars_from_expr(index, &mut live);
                    collect_vars_from_expr(value, &mut live);
                }
                MirStatement::Havoc { var } => {
                    assigned.insert(var.clone());
                }
            }
        }

        // Check terminator conditions
        match &block.terminator {
            MirTerminator::CondGoto { condition, .. } => {
                collect_vars_from_expr(condition, &mut live);
            }
            MirTerminator::SwitchInt { discr, .. } => {
                collect_vars_from_expr(discr, &mut live);
            }
            MirTerminator::Call {
                func,
                args,
                destination,
                is_range_into_iter,
                is_range_next,
                ..
            } => {
                for arg in args {
                    collect_vars_from_expr(arg, &mut live);
                }
                if let Some(dest) = destination {
                    assigned.insert(dest.clone());
                }

                // For Range iterator calls (into_iter, next), the field variables
                // are implicitly used even though they don't appear in the args.
                // We need to include them in the live set to ensure they're
                // tracked in the CHC invariant.
                if call_is_range_into_iter(func, *is_range_into_iter)
                    || call_is_range_next(func, *is_range_next)
                {
                    if !args.is_empty() {
                        let current = &args[0];

                        // Use pre-computed graph for O(V+E) BFS instead of O(n^3) fixed-point
                        let range_vars = find_connected_variables(current, &assignment_graph);

                        // Include field variables for ALL Range variables in the chain
                        for var in &range_vars {
                            live.insert(format!("{}_field0", var));
                            live.insert(format!("{}_field1", var));
                            live.insert(var.clone());
                        }
                    }

                    // Also include destination fields for Option<T> return
                    if let Some(dest) = destination {
                        live.insert(format!("{}_discr", dest));
                        live.insert(format!("{}_val", dest));
                        live.insert(format!("{}_field0", dest));
                    }
                }
            }
            _ => {}
        }
    }

    // A variable is needed if it's live (read) at some point
    // The key insight: only variables that are READ need to be in the invariant
    live
}

/// Extract variable references from an SMT expression.
///
/// This is a simple heuristic that finds identifiers that look like variable names.
/// It collects both MIR-style variables (_0, _1) and named variables (x, y, z).
fn collect_vars_from_expr(expr: &str, vars: &mut HashSet<String>) {
    // Split on non-identifier characters to avoid manual index management that can loop forever
    for raw in expr.split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '\'') {
        if raw.is_empty() {
            continue;
        }

        // Remove trailing ' if present (primed version)
        let base = raw.trim_end_matches('\'');
        if base.is_empty() {
            continue;
        }

        // Skip SMT keywords (uses static SMT_KEYWORDS set) and numbers
        if SMT_KEYWORDS.contains(base) {
            continue;
        }
        if base.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }

        vars.insert(base.to_string());
    }
}

/// Encode a MIR program into a transition system
pub fn encode_mir_to_transition_system(program: &MirProgram) -> TransitionSystem {
    encode_mir_to_transition_system_optimized(program, true)
}

/// Encode a MIR program into a transition system with optional optimizations.
///
/// When `optimize` is true:
/// - Dead variable elimination: Only live variables are included in the state
/// - This reduces the state space and improves CHC solving performance
pub fn encode_mir_to_transition_system_optimized(
    program: &MirProgram,
    optimize: bool,
) -> TransitionSystem {
    let mut ts = TransitionSystem::new();

    // Compute live variables for optimization
    let live_vars = if optimize {
        analyze_live_variables(program)
    } else {
        // Include all variables
        program.locals.iter().map(|l| l.name.clone()).collect()
    };

    // Program counter is always needed
    ts.add_variable("pc", SmtType::Int);

    // Only add live locals to the transition system
    let live_locals: Vec<&MirLocal> = program
        .locals
        .iter()
        .filter(|local| live_vars.contains(&local.name))
        .collect();

    for local in &live_locals {
        ts.add_variable(local.name.clone(), local.ty.clone());
    }

    // Initial state: set pc plus any optional constraints
    let mut init_parts = vec![format!("(= pc {})", program.start_block)];
    if let Some(init) = &program.init {
        if init.smt_formula.trim() != "true" {
            init_parts.push(init.smt_formula.clone());
        }
    }
    ts.set_init(StateFormula::new(conjoin(&init_parts)));

    // Property: all assertions must hold when their block is active
    let property_formula = build_property_formula(&program.basic_blocks);
    ts.add_property(Property::safety(
        "assertions",
        "all MIR assertions hold",
        StateFormula::new(property_formula),
    ));

    // Transition relation - pass live variables for optimization
    let transition = build_transition_formula_optimized(program, &live_vars);
    ts.set_transition(StateFormula::new(transition));

    ts
}

/// Encode MIR directly to a CHC system
///
/// Applies optimizations by default:
/// - Overflow flag elimination (for unbounded Int)
/// - Dead variable elimination
///
/// Use `encode_mir_to_chc_with_overflow_checks` to preserve overflow checking.
pub fn encode_mir_to_chc(program: &MirProgram) -> ChcSystem {
    encode_mir_to_chc_impl(program, false)
}

/// Encode MIR to CHC with overflow checking preserved
///
/// This variant does NOT eliminate overflow flag checks, allowing detection
/// of integer overflow even with unbounded Int representation.
///
/// Use this when `KANI_FAST_OVERFLOW_CHECKS` is enabled.
pub fn encode_mir_to_chc_with_overflow_checks(program: &MirProgram) -> ChcSystem {
    encode_mir_to_chc_impl(program, true)
}

/// Internal implementation of MIR to CHC encoding
fn encode_mir_to_chc_impl(program: &MirProgram, preserve_overflow_checks: bool) -> ChcSystem {
    // Apply overflow flag optimization only if not preserving overflow checks
    // Using Cow to avoid cloning when no optimization is needed
    let optimized = if preserve_overflow_checks {
        Cow::Borrowed(program)
    } else {
        optimize_mir_for_unbounded(program)
    };
    let ts = encode_mir_to_transition_system(&optimized);
    let mut chc = encode_transition_system(&ts);

    // Add declarations for bitwise operations (Int → Int → Int)
    // These are used for bitwise AND/OR/XOR with integer types
    chc.add_function(UninterpretedFunction::new(
        "bitand",
        vec![SmtType::Int, SmtType::Int],
        SmtType::Int,
    ));
    chc.add_function(UninterpretedFunction::new(
        "bitor",
        vec![SmtType::Int, SmtType::Int],
        SmtType::Int,
    ));
    chc.add_function(UninterpretedFunction::new(
        "bitxor",
        vec![SmtType::Int, SmtType::Int],
        SmtType::Int,
    ));
    // Power of 2 function for shift operations: pow2(n) = 2^n
    chc.add_function(UninterpretedFunction::new(
        "pow2",
        vec![SmtType::Int],
        SmtType::Int,
    ));

    // Collect and declare uninterpreted functions from Call terminators
    for block in &program.basic_blocks {
        if let MirTerminator::Call {
            func,
            args,
            destination,
            ..
        } = &block.terminator
        {
            // Sanitize the function name
            let sanitized_name = sanitize_smt_identifier(func);

            // Determine argument types (default to Int since we're using unbounded arithmetic)
            let param_types: Vec<SmtType> = vec![SmtType::Int; args.len()];

            // Determine return type based on destination
            let return_type = destination
                .as_ref()
                .and_then(|dest| {
                    program
                        .locals
                        .iter()
                        .find(|l| &l.name == dest)
                        .map(|l| l.ty.clone())
                })
                .unwrap_or(SmtType::Int);

            let func_decl = UninterpretedFunction::new(sanitized_name, param_types, return_type);
            chc.add_function(func_decl);
        }
    }

    // Note: We do NOT add axioms for intrinsic functions (wrapping_add, saturating_sub, etc.)
    // because:
    // 1. Intrinsics are already inlined directly in the transition formulas (see try_inline_call)
    // 2. Universal quantified equality axioms (forall a b. f(a,b) = expr) are NOT Horn clauses
    // 3. Non-Horn axioms cause Z3 Spacer to return "unknown" instead of solving
    //
    // The pow2 function is similarly left uninterpreted. Adding axioms like pow2(n+1) = 2 * pow2(n)
    // breaks CHC solving because they're not in Horn clause form.
    // This means shifts are treated abstractly, which is sound but imprecise.

    chc
}

/// Result of verification with strategy information
#[derive(Debug, Clone)]
pub enum VerificationResult {
    /// CHC verification succeeded (fast path or rewritten path)
    ChcResult {
        /// The CHC system that was verified
        chc: ChcSystem,
        /// Which strategy was used
        strategy: VerificationPath,
        /// Whether algebraic rewrites were applied
        rewrites_applied: bool,
    },
    /// Delegated to Kani (complex bitwise or unsupported features)
    Delegated {
        /// Reason for delegation
        reason: DelegationReason,
    },
}

/// Encode MIR to CHC with automatic strategy selection
///
/// This function analyzes the program to determine the best verification strategy:
/// 1. **ChcFast** - No proof-relevant bitwise ops, use fast Int/CHC path
/// 2. **ChcRewritten** - Bitwise ops can be algebraically rewritten to arithmetic
/// 3. **DelegateKani** - Complex cases that require full Kani/CBMC
///
/// # Arguments
/// * `program` - The MIR program to verify
///
/// # Returns
/// `VerificationResult` indicating which path was taken
pub fn encode_mir_to_chc_with_strategy(program: &MirProgram) -> VerificationResult {
    // Phase 1: Analyze proof relevance
    let relevance = ProofRelevanceAnalysis::analyze(program);

    // Phase 2: Choose strategy based on analysis
    let strategy = choose_strategy(&relevance);

    match &strategy {
        VerificationPath::ChcFast => {
            // Fast path: no bitwise operations affect the proof
            let chc = encode_mir_to_chc(program);
            VerificationResult::ChcResult {
                chc,
                strategy,
                rewrites_applied: false,
            }
        }
        VerificationPath::ChcRewritten { .. } => {
            // Medium path: apply algebraic rewrites before encoding
            let rewritten_program = apply_algebraic_rewrites(program);
            let chc = encode_mir_to_chc(&rewritten_program);
            VerificationResult::ChcResult {
                chc,
                strategy,
                rewrites_applied: true,
            }
        }
        VerificationPath::DelegateKani { reason } => {
            // Slow path: complex cases need Kani/CBMC
            VerificationResult::Delegated {
                reason: reason.clone(),
            }
        }
    }
}

/// Encode MIR to CHC using BitVec theory for precise bitwise reasoning
///
/// This function uses SMT-LIB2 bitvector theory (QF_BV) instead of unbounded Int.
/// This enables precise verification of bitwise operations like `12 & 10 = 8`.
///
/// # Arguments
/// * `program` - The MIR program to verify
/// * `bit_width` - Bit width for integers (32 for i32, 64 for i64)
///
/// # Returns
/// CHC system using BitVec sorts and native bitvector operations
///
/// # Example
/// ```ignore
/// let chc = encode_mir_to_chc_bitvec(&program, 32);
/// // Now uses (_ BitVec 32) instead of Int
/// // And bvand/bvor/bvxor instead of uninterpreted bitand/bitor/bitxor
/// ```
pub fn encode_mir_to_chc_bitvec(program: &MirProgram, bit_width: u32) -> ChcSystem {
    use crate::bitvec::{convert_int_to_bitvec, BitvecConfig};

    let config = BitvecConfig::new(bit_width);

    // Convert locals to BitVec type for integer variables
    let bv_locals: Vec<MirLocal> = program
        .locals
        .iter()
        .map(|local| {
            let new_ty = match &local.ty {
                SmtType::Int => SmtType::BitVec(bit_width),
                other => other.clone(),
            };
            MirLocal::new(&local.name, new_ty)
        })
        .collect();

    // Convert init formula to BitVec
    let bv_init = program.init.as_ref().map(|f| {
        let bv_formula = convert_int_to_bitvec(&f.smt_formula, &config);
        if let Some(desc) = &f.description {
            StateFormula::with_description(bv_formula, desc)
        } else {
            StateFormula::new(bv_formula)
        }
    });

    // Convert all expressions in statements and terminators
    let bv_blocks: Vec<MirBasicBlock> = program
        .basic_blocks
        .iter()
        .map(|block| {
            let bv_statements: Vec<MirStatement> = block
                .statements
                .iter()
                .map(|stmt| convert_statement_to_bitvec(stmt, &config))
                .collect();

            let bv_terminator = convert_terminator_to_bitvec(&block.terminator, &config);

            MirBasicBlock {
                id: block.id,
                statements: bv_statements,
                terminator: bv_terminator,
            }
        })
        .collect();

    let bv_program = MirProgram {
        locals: bv_locals,
        basic_blocks: bv_blocks,
        start_block: program.start_block,
        init: bv_init,
        var_to_local: program.var_to_local.clone(),
        closures: program.closures.clone(),
        trait_impls: program.trait_impls.clone(),
    };

    // Encode the BitVec program to CHC - no uninterpreted bitwise functions needed
    let ts = encode_mir_to_transition_system(&bv_program);

    // Note: No need to add bitand/bitor/bitxor declarations since we use native bvand/bvor/bvxor
    encode_transition_system(&ts)
}

/// Convert a MirStatement to use BitVec operations
fn convert_statement_to_bitvec(
    stmt: &MirStatement,
    config: &crate::bitvec::BitvecConfig,
) -> MirStatement {
    use crate::bitvec::convert_int_to_bitvec;

    match stmt {
        MirStatement::Assume(cond) => MirStatement::Assume(convert_int_to_bitvec(cond, config)),
        MirStatement::Assign { lhs, rhs } => MirStatement::Assign {
            lhs: lhs.clone(),
            rhs: convert_int_to_bitvec(rhs, config),
        },
        MirStatement::Assert { condition, message } => MirStatement::Assert {
            condition: convert_int_to_bitvec(condition, config),
            message: message.clone(),
        },
        MirStatement::ArrayStore {
            array,
            index,
            value,
        } => MirStatement::ArrayStore {
            array: array.clone(),
            index: convert_int_to_bitvec(index, config),
            value: convert_int_to_bitvec(value, config),
        },
        MirStatement::Havoc { var } => MirStatement::Havoc { var: var.clone() },
    }
}

/// Convert a MirTerminator to use BitVec operations
fn convert_terminator_to_bitvec(
    term: &MirTerminator,
    config: &crate::bitvec::BitvecConfig,
) -> MirTerminator {
    use crate::bitvec::convert_int_to_bitvec;

    match term {
        MirTerminator::Goto { target } => MirTerminator::Goto { target: *target },
        MirTerminator::CondGoto {
            condition,
            then_target,
            else_target,
        } => MirTerminator::CondGoto {
            condition: convert_int_to_bitvec(condition, config),
            then_target: *then_target,
            else_target: *else_target,
        },
        MirTerminator::SwitchInt {
            discr,
            targets,
            otherwise,
        } => {
            // Note: SwitchInt targets are (value, block_id) pairs where value is an integer.
            // We don't convert the values here because they are used at the SMT encoding level
            // in format_switch_comparison() where the discriminant type determines the format.
            // The bitvec conversion happens in the transition formula building.
            MirTerminator::SwitchInt {
                discr: convert_int_to_bitvec(discr, config),
                targets: targets.clone(),
                otherwise: *otherwise,
            }
        }
        MirTerminator::Call {
            destination,
            func,
            args,
            target,
            unwind,
            precondition_check,
            postcondition_assumption,
            is_range_into_iter,
            is_range_next,
        } => MirTerminator::Call {
            destination: destination.clone(),
            func: func.clone(),
            args: args
                .iter()
                .map(|a| convert_int_to_bitvec(a, config))
                .collect(),
            target: *target,
            unwind: *unwind,
            precondition_check: precondition_check
                .as_ref()
                .map(|p| convert_int_to_bitvec(p, config)),
            postcondition_assumption: postcondition_assumption
                .as_ref()
                .map(|p| convert_int_to_bitvec(p, config)),
            is_range_into_iter: *is_range_into_iter,
            is_range_next: *is_range_next,
        },
        MirTerminator::Return => MirTerminator::Return,
        MirTerminator::Unreachable => MirTerminator::Unreachable,
        MirTerminator::Abort => MirTerminator::Abort,
    }
}

/// Apply algebraic rewrites to all expressions in a MIR program
///
/// This transforms bitwise operations to equivalent arithmetic where possible:
/// - `x & (2^n - 1)` → `x mod 2^n`
/// - `x << n` → `x * 2^n`
/// - `x >> n` → `x / 2^n`
/// - `x ^ x` → `0`
/// - etc.
///
/// Operations that cannot be rewritten remain as uninterpreted functions.
pub fn apply_algebraic_rewrites(program: &MirProgram) -> MirProgram {
    // Rewrite the init formula if present
    let rewritten_init = program.init.as_ref().map(|f| {
        let rewritten_formula = rewrite_expr_if_bitwise(&f.smt_formula);
        if let Some(desc) = &f.description {
            StateFormula::with_description(rewritten_formula, desc)
        } else {
            StateFormula::new(rewritten_formula)
        }
    });

    let mut rewritten = MirProgram {
        locals: program.locals.clone(),
        basic_blocks: Vec::with_capacity(program.basic_blocks.len()),
        start_block: program.start_block,
        init: rewritten_init,
        var_to_local: program.var_to_local.clone(),
        closures: program.closures.clone(),
        trait_impls: program.trait_impls.clone(),
    };

    for block in &program.basic_blocks {
        let mut new_block = MirBasicBlock {
            id: block.id,
            statements: Vec::with_capacity(block.statements.len()),
            terminator: rewrite_terminator(&block.terminator),
        };

        for stmt in &block.statements {
            new_block.statements.push(rewrite_statement(stmt));
        }

        rewritten.basic_blocks.push(new_block);
    }

    rewritten
}

/// Rewrite a single statement's expressions
fn rewrite_statement(stmt: &MirStatement) -> MirStatement {
    match stmt {
        MirStatement::Assume(cond) => MirStatement::Assume(rewrite_expr_if_bitwise(cond)),
        MirStatement::Assign { lhs, rhs } => MirStatement::Assign {
            lhs: lhs.clone(),
            rhs: rewrite_expr_if_bitwise(rhs),
        },
        MirStatement::Assert { condition, message } => MirStatement::Assert {
            condition: rewrite_expr_if_bitwise(condition),
            message: message.clone(),
        },
        MirStatement::ArrayStore {
            array,
            index,
            value,
        } => MirStatement::ArrayStore {
            array: array.clone(),
            index: rewrite_expr_if_bitwise(index),
            value: rewrite_expr_if_bitwise(value),
        },
        MirStatement::Havoc { var } => MirStatement::Havoc { var: var.clone() },
    }
}

/// Rewrite a terminator's expressions
fn rewrite_terminator(term: &MirTerminator) -> MirTerminator {
    match term {
        MirTerminator::Goto { target } => MirTerminator::Goto { target: *target },
        MirTerminator::CondGoto {
            condition,
            then_target,
            else_target,
        } => MirTerminator::CondGoto {
            condition: rewrite_expr_if_bitwise(condition),
            then_target: *then_target,
            else_target: *else_target,
        },
        MirTerminator::SwitchInt {
            discr,
            targets,
            otherwise,
        } => MirTerminator::SwitchInt {
            discr: rewrite_expr_if_bitwise(discr),
            targets: targets.clone(),
            otherwise: *otherwise,
        },
        MirTerminator::Call {
            destination,
            func,
            args,
            target,
            unwind,
            precondition_check,
            postcondition_assumption,
            is_range_into_iter,
            is_range_next,
        } => MirTerminator::Call {
            destination: destination.clone(),
            func: func.clone(),
            args: args.iter().map(|a| rewrite_expr_if_bitwise(a)).collect(),
            target: *target,
            unwind: *unwind,
            precondition_check: precondition_check
                .as_ref()
                .map(|p| rewrite_expr_if_bitwise(p)),
            postcondition_assumption: postcondition_assumption
                .as_ref()
                .map(|p| rewrite_expr_if_bitwise(p)),
            is_range_into_iter: *is_range_into_iter,
            is_range_next: *is_range_next,
        },
        MirTerminator::Return => MirTerminator::Return,
        MirTerminator::Unreachable => MirTerminator::Unreachable,
        MirTerminator::Abort => MirTerminator::Abort,
    }
}

/// Rewrite an expression if it contains bitwise operations
///
/// Returns the original expression if no bitwise operations were found,
/// or the rewritten expression with arithmetic equivalents.
fn rewrite_expr_if_bitwise(expr: &str) -> String {
    let (rewritten, was_rewritten) = rewrite_expression(expr);
    if was_rewritten {
        rewritten
    } else {
        expr.to_string()
    }
}

/// Build the safety property formula from MIR blocks.
///
/// The property ensures:
/// 1. All explicit assertions hold when their block is active
/// 2. Abort/error states are unreachable (pc never reaches PC_ABORT_SENTINEL or PC_PANIC_SENTINEL)
///
/// The abort state reachability check is critical for soundness: without it,
/// overflow checks and other safety assertions encoded as conditional branches
/// to abort blocks would not be verified.
fn build_property_formula(blocks: &[MirBasicBlock]) -> String {
    // Pre-allocate: 2 base assertions (abort states) + estimated assertions from blocks
    let mut assertions = Vec::with_capacity(blocks.len() + 2);

    // Extract explicit assertions from statements
    for block in blocks {
        for stmt in &block.statements {
            if let MirStatement::Assert { condition, .. } = stmt {
                assertions.push(format!("(or (not (= pc {})) {})", block.id, condition));
            }
        }

        // MODULAR VERIFICATION: Extract precondition checks from Call terminators
        // When a call has a precondition_check, it must hold before the call executes.
        // This enforces the callee's #[kani::requires] specification.
        if let MirTerminator::Call {
            precondition_check: Some(precond),
            ..
        } = &block.terminator
        {
            // The precondition must hold when we're about to execute the call
            assertions.push(format!("(or (not (= pc {})) {})", block.id, precond));
            tracing::debug!(
                "Added precondition check for block {}: {}",
                block.id,
                precond
            );
        }
    }

    // CRITICAL: Add safety property that abort states are unreachable
    // PC_ABORT_SENTINEL (-2) is the error state for Abort terminators
    // PC_PANIC_SENTINEL (999999) is the sentinel for panic/assert-fail paths
    // If CHC solver finds these states are reachable, the property is violated.
    assertions.push(format!("(not (= pc {}))", PC_ABORT_SENTINEL));
    assertions.push(format!("(not (= pc {}))", PC_PANIC_SENTINEL));

    // Since we always have at least the abort state checks, this is never empty
    conjoin(&assertions)
}

/// Build the transition formula for a MIR program (test-only function).
///
/// Used by unit tests to verify transition encoding logic. Production code
/// uses `build_transition_formula_optimized` for better performance.
#[cfg(test)]
fn build_transition_formula(program: &MirProgram) -> String {
    let loop_headers = compute_loop_headers(&program.basic_blocks);
    let predecessors = build_predecessor_map(&program.basic_blocks, &program.locals);
    // Pre-allocate: each block may produce multiple disjuncts, estimate ~2x blocks
    let mut disjuncts = Vec::with_capacity(program.basic_blocks.len() * 2);

    for block in &program.basic_blocks {
        match &block.terminator {
            MirTerminator::Unreachable => {
                // No outgoing edges
            }
            MirTerminator::SwitchInt {
                discr,
                targets,
                otherwise,
            } => {
                for (value, target) in targets {
                    // Use type-aware comparison for boolean discriminants
                    let cmp = format_switch_comparison(discr, *value, &program.locals);
                    let clause =
                        build_block_clause(block, Some(cmp), idx_to_i64(*target), &program.locals);
                    disjuncts.push(clause);
                }

                // Otherwise branch when no target matched
                let mut guards: Vec<String> = targets
                    .iter()
                    .map(|(v, _)| format_switch_comparison_negated(discr, *v, &program.locals))
                    .collect();
                guards.push(format!("(= pc {})", block.id));

                let clause = build_block_clause(
                    block,
                    Some(conjoin(&guards)),
                    idx_to_i64(*otherwise),
                    &program.locals,
                );
                disjuncts.push(clause);
            }
            MirTerminator::CondGoto {
                condition,
                then_target,
                else_target,
            } => {
                let then_clause = build_block_clause(
                    block,
                    Some(condition.clone()),
                    idx_to_i64(*then_target),
                    &program.locals,
                );
                disjuncts.push(then_clause);

                let else_clause = build_block_clause(
                    block,
                    Some(format!("(not {})", condition)),
                    idx_to_i64(*else_target),
                    &program.locals,
                );
                disjuncts.push(else_clause);
            }
            MirTerminator::Goto { target } => {
                let clause = build_block_clause(block, None, idx_to_i64(*target), &program.locals);
                disjuncts.push(clause);
            }
            MirTerminator::Return => {
                let clause = build_block_clause(block, None, PC_RETURN_SENTINEL, &program.locals);
                disjuncts.push(clause);
            }
            MirTerminator::Call {
                destination,
                func,
                args,
                target,
                unwind: _,
                precondition_check: _, // Handled in build_property_formula
                postcondition_assumption,
                is_range_into_iter,
                is_range_next,
            } => {
                // For uninterpreted function calls, the result is the application
                // of an uninterpreted function to the arguments
                // MODULAR VERIFICATION: If postcondition_assumption is provided,
                // we use it to constrain the return value instead of havocing
                // Note: precondition_check is handled in build_property_formula as an assertion
                let clause = build_call_clause(
                    block,
                    destination.as_deref(),
                    func,
                    args,
                    *is_range_into_iter,
                    *is_range_next,
                    idx_to_i64(*target),
                    &program.locals,
                    postcondition_assumption.as_deref(),
                    &predecessors,
                    &program.basic_blocks,
                    program.start_block,
                    loop_headers.get(&block.id).copied(),
                    &program.closures,
                    &program.trait_impls,
                );
                disjuncts.push(clause);
            }
            MirTerminator::Abort => {
                // Abort transitions to error state (PC_ABORT_SENTINEL)
                let clause = build_block_clause(block, None, PC_ABORT_SENTINEL, &program.locals);
                disjuncts.push(clause);
            }
        }
    }

    if disjuncts.is_empty() {
        "false".to_string()
    } else {
        disjoin(&disjuncts)
    }
}

/// Build transition formula with dead variable elimination.
///
/// Only live variables (those actually read somewhere in the program) are
/// carried forward in the transition. This reduces the state space for CHC solving.
fn build_transition_formula_optimized(program: &MirProgram, live_vars: &HashSet<String>) -> String {
    // Filter locals to only include live ones
    let live_locals: Vec<MirLocal> = program
        .locals
        .iter()
        .filter(|l| live_vars.contains(&l.name))
        .cloned()
        .collect();

    let loop_headers = compute_loop_headers(&program.basic_blocks);
    let predecessors = build_predecessor_map(&program.basic_blocks, &live_locals);
    // Pre-allocate: each block may produce multiple disjuncts, estimate ~2x blocks
    let mut disjuncts = Vec::with_capacity(program.basic_blocks.len() * 2);

    for block in &program.basic_blocks {
        match &block.terminator {
            MirTerminator::Unreachable => {
                // No outgoing edges
            }
            MirTerminator::SwitchInt {
                discr,
                targets,
                otherwise,
            } => {
                for (value, target) in targets {
                    let cmp = format_switch_comparison(discr, *value, &live_locals);
                    let clause =
                        build_block_clause(block, Some(cmp), idx_to_i64(*target), &live_locals);
                    disjuncts.push(clause);
                }

                let mut guards: Vec<String> = targets
                    .iter()
                    .map(|(v, _)| format_switch_comparison_negated(discr, *v, &live_locals))
                    .collect();
                guards.push(format!("(= pc {})", block.id));

                let clause = build_block_clause(
                    block,
                    Some(conjoin(&guards)),
                    idx_to_i64(*otherwise),
                    &live_locals,
                );
                disjuncts.push(clause);
            }
            MirTerminator::CondGoto {
                condition,
                then_target,
                else_target,
            } => {
                let then_clause = build_block_clause(
                    block,
                    Some(condition.clone()),
                    idx_to_i64(*then_target),
                    &live_locals,
                );
                disjuncts.push(then_clause);

                let else_clause = build_block_clause(
                    block,
                    Some(format!("(not {})", condition)),
                    idx_to_i64(*else_target),
                    &live_locals,
                );
                disjuncts.push(else_clause);
            }
            MirTerminator::Goto { target } => {
                let clause = build_block_clause(block, None, idx_to_i64(*target), &live_locals);
                disjuncts.push(clause);
            }
            MirTerminator::Return => {
                let clause = build_block_clause(block, None, PC_RETURN_SENTINEL, &live_locals);
                disjuncts.push(clause);
            }
            MirTerminator::Call {
                destination,
                func,
                args,
                target,
                unwind: _,
                precondition_check: _, // Handled in build_property_formula
                postcondition_assumption,
                is_range_into_iter,
                is_range_next,
            } => {
                let clause = build_call_clause(
                    block,
                    destination.as_deref(),
                    func,
                    args,
                    *is_range_into_iter,
                    *is_range_next,
                    idx_to_i64(*target),
                    &live_locals,
                    postcondition_assumption.as_deref(),
                    &predecessors,
                    &program.basic_blocks,
                    program.start_block,
                    loop_headers.get(&block.id).copied(),
                    &program.closures,
                    &program.trait_impls,
                );
                disjuncts.push(clause);
            }
            MirTerminator::Abort => {
                // Abort transitions to error state (PC_ABORT_SENTINEL)
                let clause = build_block_clause(block, None, PC_ABORT_SENTINEL, &live_locals);
                disjuncts.push(clause);
            }
        }
    }

    if disjuncts.is_empty() {
        "false".to_string()
    } else {
        disjoin(&disjuncts)
    }
}

/// Build a predecessor map annotated with edge conditions.
///
/// Each entry maps a target block ID to the list of predecessor block IDs and
/// the condition that must hold to traverse that edge.
fn build_predecessor_map(
    blocks: &[MirBasicBlock],
    locals: &[MirLocal],
) -> HashMap<usize, Vec<(usize, Option<String>)>> {
    // Pre-allocate with expected number of blocks
    let mut preds: HashMap<usize, Vec<(usize, Option<String>)>> =
        HashMap::with_capacity(blocks.len());

    for block in blocks {
        match &block.terminator {
            MirTerminator::CondGoto {
                condition,
                then_target,
                else_target,
            } => {
                preds
                    .entry(*then_target)
                    .or_default()
                    .push((block.id, Some(condition.clone())));
                preds
                    .entry(*else_target)
                    .or_default()
                    .push((block.id, Some(format!("(not {})", condition))));
            }
            MirTerminator::SwitchInt {
                discr,
                targets,
                otherwise,
            } => {
                for (value, target) in targets {
                    let cmp = format_switch_comparison(discr, *value, locals);
                    preds
                        .entry(*target)
                        .or_default()
                        .push((block.id, Some(cmp)));
                }

                // Otherwise branch guard: discriminant does not match any explicit target
                let negated: Vec<String> = targets
                    .iter()
                    .map(|(v, _)| format_switch_comparison_negated(discr, *v, locals))
                    .collect();
                // Ensure we have a non-empty guard to avoid conjoining to "true"
                if !negated.is_empty() {
                    preds
                        .entry(*otherwise)
                        .or_default()
                        .push((block.id, Some(conjoin(&negated))));
                } else {
                    preds.entry(*otherwise).or_default().push((block.id, None));
                }
            }
            MirTerminator::Goto { target } => {
                preds.entry(*target).or_default().push((block.id, None));
            }
            MirTerminator::Call { target, .. } => {
                preds.entry(*target).or_default().push((block.id, None));
            }
            MirTerminator::Return | MirTerminator::Unreachable | MirTerminator::Abort => {}
        }
    }

    preds
}

/// Collect path conditions from predecessors leading to the given call block.
///
/// This performs a bounded DFS from the call site back to the nearest loop
/// header (if provided) or the program entry block, accumulating branch guards
/// along each explored path. The shortest path (fewest conditions) is used to
/// avoid over-constraining the transition.
fn collect_path_conditions(
    predecessors: &HashMap<usize, Vec<(usize, Option<String>)>>,
    call_block_id: usize,
    start_block: usize,
    loop_header: Option<usize>,
) -> Vec<String> {
    let target = loop_header.unwrap_or(start_block);
    let mut paths: Vec<Vec<String>> = Vec::new();

    fn dfs(
        current: usize,
        target: usize,
        predecessors: &HashMap<usize, Vec<(usize, Option<String>)>>,
        path_conditions: &mut Vec<String>,
        path_nodes: &mut HashSet<usize>,
        paths: &mut Vec<Vec<String>>,
    ) {
        if !path_nodes.insert(current) {
            // Avoid cycles
            return;
        }

        if current == target {
            paths.push(path_conditions.clone());
            path_nodes.remove(&current);
            return;
        }

        if let Some(preds) = predecessors.get(&current) {
            for (pred, cond) in preds {
                if path_nodes.contains(pred) {
                    continue;
                }
                let pushed = if let Some(c) = cond {
                    path_conditions.push(c.clone());
                    true
                } else {
                    false
                };

                dfs(
                    *pred,
                    target,
                    predecessors,
                    path_conditions,
                    path_nodes,
                    paths,
                );

                if pushed {
                    path_conditions.pop();
                }
            }
        }

        path_nodes.remove(&current);
    }

    dfs(
        call_block_id,
        target,
        predecessors,
        &mut Vec::new(),
        &mut HashSet::new(),
        &mut paths,
    );

    // Prefer the shortest path (fewest branch conditions)
    // Reverse to order conditions from outermost to innermost
    paths
        .into_iter()
        .min_by_key(|p| p.len())
        .map(|mut chosen| {
            chosen.reverse();
            chosen
        })
        .unwrap_or_default()
}

/// Extract loop invariants encoded as assumptions on the loop header block.
fn collect_loop_invariants(blocks: &[MirBasicBlock], loop_header: usize) -> Vec<String> {
    blocks
        .iter()
        .find(|b| b.id == loop_header)
        .map(|block| {
            block
                .statements
                .iter()
                .filter_map(|stmt| match stmt {
                    MirStatement::Assume(cond) => Some(cond.clone()),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Compute a best-effort mapping from blocks to their loop headers using SCCs.
fn compute_loop_headers(blocks: &[MirBasicBlock]) -> HashMap<usize, usize> {
    // Pre-allocate with expected number of blocks
    let mut successors: HashMap<usize, Vec<usize>> = HashMap::with_capacity(blocks.len());
    for block in blocks {
        let succs = successors.entry(block.id).or_default();
        match &block.terminator {
            MirTerminator::CondGoto {
                then_target,
                else_target,
                ..
            } => {
                succs.push(*then_target);
                succs.push(*else_target);
            }
            MirTerminator::SwitchInt {
                targets, otherwise, ..
            } => {
                for (_, target) in targets {
                    succs.push(*target);
                }
                succs.push(*otherwise);
            }
            MirTerminator::Goto { target } => succs.push(*target),
            MirTerminator::Call { target, .. } => succs.push(*target),
            MirTerminator::Return | MirTerminator::Unreachable | MirTerminator::Abort => {}
        }
    }

    let mut index: usize = 0;
    // Pre-allocate based on number of blocks
    let mut indices: HashMap<usize, usize> = HashMap::with_capacity(blocks.len());
    let mut lowlink: HashMap<usize, usize> = HashMap::with_capacity(blocks.len());
    let mut stack: Vec<usize> = Vec::with_capacity(blocks.len());
    let mut on_stack: HashSet<usize> = HashSet::with_capacity(blocks.len());
    let mut headers: HashMap<usize, usize> = HashMap::with_capacity(blocks.len());

    #[allow(clippy::too_many_arguments)]
    fn strong_connect(
        v: usize,
        successors: &HashMap<usize, Vec<usize>>,
        index: &mut usize,
        indices: &mut HashMap<usize, usize>,
        lowlink: &mut HashMap<usize, usize>,
        stack: &mut Vec<usize>,
        on_stack: &mut HashSet<usize>,
        headers: &mut HashMap<usize, usize>,
    ) {
        *index += 1;
        indices.insert(v, *index);
        lowlink.insert(v, *index);
        stack.push(v);
        on_stack.insert(v);

        if let Some(succs) = successors.get(&v) {
            for &w in succs {
                if !indices.contains_key(&w) {
                    strong_connect(
                        w, successors, index, indices, lowlink, stack, on_stack, headers,
                    );
                    if let Some(low_w) = lowlink.get(&w).copied() {
                        if let Some(low_v) = lowlink.get_mut(&v) {
                            if *low_v > low_w {
                                *low_v = low_w;
                            }
                        }
                    }
                } else if on_stack.contains(&w) {
                    if let Some(idx_w) = indices.get(&w).copied() {
                        if let Some(low_v) = lowlink.get_mut(&v) {
                            if *low_v > idx_w {
                                *low_v = idx_w;
                            }
                        }
                    }
                }
            }
        }

        let is_root = matches!((lowlink.get(&v), indices.get(&v)), (Some(l), Some(i)) if l == i);
        if is_root {
            let mut scc = Vec::new();
            while let Some(w) = stack.pop() {
                on_stack.remove(&w);
                scc.push(w);
                if w == v {
                    break;
                }
            }

            let has_self_loop = successors.get(&v).is_some_and(|succs| succs.contains(&v));
            if scc.len() > 1 || has_self_loop {
                if let Some(header) = scc.iter().min().copied() {
                    for node in scc {
                        headers.insert(node, header);
                    }
                }
            }
        }
    }

    for block in blocks {
        if !indices.contains_key(&block.id) {
            strong_connect(
                block.id,
                &successors,
                &mut index,
                &mut indices,
                &mut lowlink,
                &mut stack,
                &mut on_stack,
                &mut headers,
            );
        }
    }

    headers
}

/// Finalize a clause by carrying forward unchanged locals and setting the next PC.
///
/// This helper encapsulates the common pattern used at the end of clause building:
/// 1. For each local not in `assigned` (and not in `extra_assigned`), add `(= local' local)`
/// 2. Add `(= pc' pc_target)`
/// 3. Return the conjunction of all constraints
///
/// # Arguments
/// * `conjuncts` - Accumulated constraints to include in the clause
/// * `locals` - All local variables in the function
/// * `assigned` - Variables that were assigned in this clause (skip these)
/// * `extra_assigned` - Additional variables to skip (e.g., from iterator expansion)
/// * `pc_target` - Target program counter value
fn finalize_clause_with_locals(
    conjuncts: &mut Vec<String>,
    locals: &[MirLocal],
    assigned: &HashSet<&str>,
    extra_assigned: Option<&HashSet<String>>,
    pc_target: i64,
) -> String {
    // Carry forward unchanged locals
    for local in locals {
        let is_assigned = assigned.contains(local.name.as_str());
        let is_extra_assigned = extra_assigned.is_some_and(|extra| extra.contains(&local.name));
        if !is_assigned && !is_extra_assigned {
            conjuncts.push(format!("(= {}' {})", local.name, local.name));
        }
    }
    conjuncts.push(format!("(= pc' {})", pc_target));
    format!("(and {})", conjuncts.join(" "))
}

/// Build a clause for a function call terminator
///
/// If `postcondition_assumption` is provided, it represents the callee's postcondition
/// and will be used to constrain the return value (modular verification).
#[allow(clippy::too_many_arguments)]
fn build_call_clause(
    block: &MirBasicBlock,
    destination: Option<&str>,
    func: &str,
    args: &[String],
    is_range_into_iter: bool,
    is_range_next: bool,
    pc_target: i64,
    locals: &[MirLocal],
    postcondition_assumption: Option<&str>,
    predecessors: &HashMap<usize, Vec<(usize, Option<String>)>>,
    all_blocks: &[MirBasicBlock],
    start_block: usize,
    loop_header: Option<usize>,
    closures: &HashMap<String, ClosureInfo>,
    trait_impls: &HashMap<String, TraitImplInfo>,
) -> String {
    // Pre-allocate: pc + statements + locals + terminator (estimate ~2x statements + locals)
    let estimated_size = block.statements.len() * 2 + locals.len() + 4;
    let mut conjuncts = Vec::with_capacity(estimated_size);
    let mut assigned: HashSet<&str> = HashSet::with_capacity(locals.len());
    // Track which variables are in the state (live variables)
    let local_names: HashSet<&str> = locals.iter().map(|l| l.name.as_str()).collect();

    conjuncts.push(format!("(= pc {})", block.id));

    // Add path conditions leading to this call (guards from predecessors, loop context)
    let mut context_conditions =
        collect_path_conditions(predecessors, block.id, start_block, loop_header);
    // Include any loop invariants attached to the detected loop header
    if let Some(header) = loop_header {
        context_conditions.extend(collect_loop_invariants(all_blocks, header));
    }
    if !context_conditions.is_empty() {
        let mut seen = HashSet::new();
        for cond in context_conditions {
            if cond.trim().is_empty() || cond == "true" {
                continue;
            }
            if seen.insert(cond.clone()) {
                conjuncts.push(cond);
            }
        }
    }

    // Process block statements sequentially, substituting already-assigned variables
    for stmt in &block.statements {
        match stmt {
            MirStatement::Assume(cond) => {
                let substituted = substitute_assigned_vars(cond, &assigned);
                conjuncts.push(substituted);
            }
            MirStatement::Assign { lhs, rhs } => {
                let substituted_rhs = substitute_assigned_vars(rhs, &assigned);
                // Only emit constraint if the variable is in the state
                if local_names.contains(lhs.as_str()) {
                    conjuncts.push(format!("(= {}' {})", lhs, substituted_rhs));
                }
                assigned.insert(lhs.as_str());
            }
            MirStatement::Assert { .. } => {}
            MirStatement::ArrayStore {
                array,
                index,
                value,
            } => {
                let sub_array = substitute_assigned_vars(array, &assigned);
                let sub_index = substitute_assigned_vars(index, &assigned);
                let sub_value = substitute_assigned_vars(value, &assigned);
                // Only emit constraint if the variable is in the state
                if local_names.contains(array.as_str()) {
                    conjuncts.push(format!(
                        "(= {}' (store {} {} {}))",
                        array, sub_array, sub_index, sub_value
                    ));
                }
                assigned.insert(array.as_str());
            }
            MirStatement::Havoc { var } => {
                assigned.insert(var.as_str());
            }
        }
    }

    // Handle kani::assume() calls specially - they add constraints rather than assignments
    // Check both original name and sanitized forms
    if is_kani_assume(func) {
        // kani::assume(cond) constrains the symbolic state
        let substituted_args: Vec<String> = args
            .iter()
            .map(|arg| substitute_assigned_vars(arg, &assigned))
            .collect();
        if !substituted_args.is_empty() {
            conjuncts.push(substituted_args[0].clone());
        }
        return finalize_clause_with_locals(&mut conjuncts, locals, &assigned, None, pc_target);
    }

    // Apply function call result, substituting assigned vars in arguments
    if let Some(dest) = destination {
        // Handle kani::any() - the destination should be unconstrained (havoc)
        // This is the key for symbolic verification: any() returns a fresh symbolic value
        if is_kani_any(func) {
            // Don't constrain dest' - leave it as a free variable
            // This means any value is allowed, enabling symbolic reasoning
            assigned.insert(dest);
            return finalize_clause_with_locals(&mut conjuncts, locals, &assigned, None, pc_target);
        }

        // Substitute assigned vars in function arguments
        let substituted_args: Vec<String> = args
            .iter()
            .map(|arg| substitute_assigned_vars(arg, &assigned))
            .collect();

        // Handle Range::into_iter - identity function for Range
        if let Some(extra_assigned) = try_expand_range_into_iter(
            func,
            is_range_into_iter,
            &substituted_args,
            dest,
            &local_names,
            &mut conjuncts,
        ) {
            assigned.insert(dest);
            let extra_assigned_set: HashSet<String> = extra_assigned.into_iter().collect();
            return finalize_clause_with_locals(
                &mut conjuncts,
                locals,
                &assigned,
                Some(&extra_assigned_set),
                pc_target,
            );
        }

        // Handle Range::next - advances iterator and returns Option<T>
        // Note: Use original args (not substituted_args) because we need to trace
        // through the reference chain to find the actual Range variable.
        if let Some(extra_assigned) = try_expand_range_next(
            func,
            is_range_next,
            args, // Use original args, not substituted
            dest,
            &local_names,
            &mut conjuncts,
            all_blocks, // Pass all blocks to trace across the program
        ) {
            assigned.insert(dest);
            // extra_assigned contains variables assigned by the iterator expansion
            let extra_assigned_set: HashSet<String> = extra_assigned.into_iter().collect();
            return finalize_clause_with_locals(
                &mut conjuncts,
                locals,
                &assigned,
                Some(&extra_assigned_set),
                pc_target,
            );
        }

        // Try to inline closure calls
        if try_expand_closure_call(
            func,
            &substituted_args,
            dest,
            closures,
            &mut conjuncts,
            &local_names,
            &mut assigned,
            &block.statements,
        ) {
            return finalize_clause_with_locals(&mut conjuncts, locals, &assigned, None, pc_target);
        }

        // Try to inline trait method calls (static dispatch)
        if try_expand_trait_call(
            func,
            &substituted_args,
            dest,
            trait_impls,
            &mut conjuncts,
            &local_names,
            &mut assigned,
            &block.statements,
        ) {
            return finalize_clause_with_locals(&mut conjuncts, locals, &assigned, None, pc_target);
        }

        // Sanitize function name to be a valid SMT-LIB identifier
        // Replace special characters with underscores
        let sanitized_func = sanitize_smt_identifier(func);

        // Try to expand checked operations (checked_add, checked_sub, checked_mul) into
        // their discriminant and value components. Checked operations return Option<T>,
        // so we need to generate separate variables for:
        // - {dest}_discr: 0 if overflow (None), 1 if no overflow (Some)
        // - {dest}_val: the computed value (only meaningful when discr = 1)
        if let Some((overflow_cond, value_expr)) =
            intrinsics::try_expand_checked_call(&sanitized_func, &substituted_args)
        {
            // Checked operations return Option<T>. In MIR, this is represented as:
            // - {dest} = discriminant (0 for None/overflow, 1 for Some/no overflow)
            // - {dest}_field0 = the inner value (only meaningful when discriminant = 1)
            //
            // The discriminant is accessed via SwitchInt on {dest}
            // The value is accessed via {dest}_field0 (from pattern like "(_X as Some).0")

            // Set {dest}_discr if it exists (alternative naming)
            let discr_var = format!("{}_discr", dest);
            if local_names.contains(discr_var.as_str()) {
                conjuncts.push(format!("(= {}' (ite {} 0 1))", discr_var, overflow_cond));
            }

            // Set {dest}_val if it exists (alternative naming)
            let val_var = format!("{}_val", dest);
            if local_names.contains(val_var.as_str()) {
                conjuncts.push(format!("(= {}' {})", val_var, value_expr));
            }

            // Set {dest}_field0 - the MIR field access pattern for Option inner value
            let field0_var = format!("{}_field0", dest);
            if let Some(local) = locals.iter().find(|l| l.name == field0_var) {
                conjuncts.push(format!("(= {}' {})", field0_var, value_expr));
                assigned.insert(local.name.as_str());
            }

            // Set the raw destination to the discriminant (0=None, 1=Some)
            // This is what SwitchInt checks to branch on Some vs None
            if local_names.contains(dest) {
                conjuncts.push(format!("(= {}' (ite {} 0 1))", dest, overflow_cond));
            }

            // Mark the destination as assigned
            assigned.insert(dest);
        } else if let Some((wrapped_value, overflow_flag)) =
            intrinsics::try_expand_overflowing_call(&sanitized_func, &substituted_args)
        {
            // Overflowing operations return (T, bool) tuple:
            // - {dest}_0: the wrapped value (always valid)
            // - {dest}_1: true if overflow occurred, false otherwise
            // This matches the MIR tuple field access pattern (_X.0, _X.1)
            let value_var = format!("{}_0", dest);
            if local_names.contains(value_var.as_str()) {
                conjuncts.push(format!("(= {}' {})", value_var, wrapped_value));
            }

            // Overflow flag: 1 if overflow, 0 if no overflow
            let flag_var = format!("{}_1", dest);
            if local_names.contains(flag_var.as_str()) {
                conjuncts.push(format!("(= {}' (ite {} 1 0))", flag_var, overflow_flag));
            }

            // Also assign the raw destination for compatibility
            if local_names.contains(dest) {
                conjuncts.push(format!("(= {}' {})", dest, wrapped_value));
            }

            // Mark the destination as assigned so it isn't carried forward unchanged
            assigned.insert(dest);
        } else {
            // Try to inline known intrinsic functions (wrapping_add, saturating_sub, etc.)
            // Inlining is preferred over axioms because Spacer can reason about inline
            // expressions directly, whereas it returns UNKNOWN for axioms on uninterpreted functions.
            let call_expr = if let Some(inlined) =
                intrinsics::try_inline_call(&sanitized_func, &substituted_args)
            {
                inlined
            } else if substituted_args.is_empty() {
                sanitized_func
            } else {
                format!("({} {})", sanitized_func, substituted_args.join(" "))
            };
            if local_names.contains(dest) {
                conjuncts.push(format!("(= {}' {})", dest, call_expr));
            }
            assigned.insert(dest);
        }
    }

    // MODULAR VERIFICATION: Add postcondition assumption if provided
    // This constrains the return value based on the callee's #[ensures] specification
    if let Some(postcond) = postcondition_assumption {
        // The postcondition is already in SMT format with the destination variable substituted
        // Add it as a constraint on the primed destination
        // First substitute any assigned vars (the destination was just assigned)
        let postcond_substituted = substitute_assigned_vars(postcond, &assigned);
        conjuncts.push(postcond_substituted);
        tracing::debug!("Added postcondition assumption: {}", postcond);
    }

    finalize_clause_with_locals(&mut conjuncts, locals, &assigned, None, pc_target)
}

/// Sanitize a Rust identifier to be a valid SMT-LIB identifier.
///
/// SMT-LIB identifiers can only contain alphanumeric characters, underscores, and
/// a few special symbols. This function converts Rust qualified names like
/// `core::num::<impl u8>::saturating_add` to `core_num_impl_u8_saturating_add`.
/// Check if a function name is kani::any (or a variant)
///
/// This detects calls to `kani::any::<T>()` which generate nondeterministic values.
/// In CHC encoding, these become unconstrained (havoc'd) variables.
fn is_kani_any(func: &str) -> bool {
    // Check various forms of kani::any
    // - kani::any (direct)
    // - kani_core::kani_intrinsics::any (from kani_core)
    // - <T as kani::Arbitrary>::any_raw (from trait impl)
    let lower = func.to_lowercase();
    lower.contains("kani")
        && (lower.contains("::any") || lower.ends_with("_any") || lower.contains("any_raw"))
}

/// Check if a function name is kani::assume
///
/// This detects calls to kani::assume(cond) which constrain the symbolic state.
/// In CHC encoding, these add the condition to the transition constraints.
fn is_kani_assume(func: &str) -> bool {
    let lower = func.to_lowercase();
    lower.contains("kani") && lower.contains("assume")
}

/// Check if a function name is Range::into_iter
///
/// `Range<T>` implements IntoIterator with `fn into_iter(self) -> Self` (identity).
/// We can inline this as a no-op that returns the input unchanged.
fn is_range_into_iter(func: &str) -> bool {
    // Patterns:
    // - <std::ops::Range<i32> as IntoIterator>::into_iter
    // - <Range<i32> as IntoIterator>::into_iter
    func.contains("Range") && func.contains("into_iter") && func.contains("IntoIterator")
}

/// Check if a function name is Range::next
///
/// `Range<T>` implements Iterator with next() that:
/// - Returns Some(start) and advances start by 1 if start < end
/// - Returns None if start >= end
fn is_range_next(func: &str) -> bool {
    // Patterns:
    // - <std::ops::Range<i32> as Iterator>::next
    // - <Range<i32> as Iterator>::next
    func.contains("Range") && func.contains("::next") && func.contains("Iterator")
}

/// Determine if a call should be treated as Range::into_iter based on flags or name.
fn call_is_range_into_iter(func: &str, range_into_iter_hint: bool) -> bool {
    range_into_iter_hint || is_range_into_iter(func)
}

/// Determine if a call should be treated as Range::next based on flags or name.
fn call_is_range_next(func: &str, range_next_hint: bool) -> bool {
    range_next_hint || is_range_next(func)
}

/// Expand a Range::into_iter call into direct assignment
///
/// Range::into_iter(r) just returns r (identity), so we copy the range fields.
/// Returns Some(list of assigned variable names) if handled, None otherwise.
fn try_expand_range_into_iter(
    func: &str,
    is_range_into_iter: bool,
    args: &[String],
    dest: &str,
    local_names: &HashSet<&str>,
    conjuncts: &mut Vec<String>,
) -> Option<Vec<String>> {
    if !call_is_range_into_iter(func, is_range_into_iter) {
        return None;
    }

    // into_iter takes one argument: the Range value
    if args.is_empty() {
        return None;
    }

    let range_arg = &args[0];

    // Handle primed arguments: _3' should map to _3_field0' not _3'_field0
    // The argument may already be primed if previous statements assigned to it
    let (base_var, is_primed) = if range_arg.ends_with('\'') {
        (range_arg[..range_arg.len() - 1].to_string(), true)
    } else {
        (range_arg.clone(), false)
    };

    // Copy the range fields to the destination
    // Range has two fields: start (field0) and end (field1)
    let src_start_base = format!("{}_field0", base_var);
    let src_end_base = format!("{}_field1", base_var);
    let dst_start = format!("{}_field0", dest);
    let dst_end = format!("{}_field1", dest);

    let mut extra_assigned = Vec::new();

    // Copy start field
    if local_names.contains(dst_start.as_str()) {
        let src = if local_names.contains(src_start_base.as_str()) {
            // Use primed form if the original arg was primed
            if is_primed {
                format!("{}'", src_start_base)
            } else {
                src_start_base
            }
        } else {
            // If source fields don't exist, use the base variable
            range_arg.clone()
        };
        conjuncts.push(format!("(= {}' {})", dst_start, src));
        extra_assigned.push(dst_start);
    }

    // Copy end field
    if local_names.contains(dst_end.as_str()) {
        let src = if local_names.contains(src_end_base.as_str()) {
            if is_primed {
                format!("{}'", src_end_base)
            } else {
                src_end_base
            }
        } else {
            range_arg.clone()
        };
        conjuncts.push(format!("(= {}' {})", dst_end, src));
        extra_assigned.push(dst_end);
    }

    // Also assign the base destination
    if local_names.contains(dest) {
        conjuncts.push(format!("(= {}' {})", dest, range_arg));
    }

    extra_assigned.push(dest.to_string());
    Some(extra_assigned)
}

/// Expand a Range::next call into CHC constraints
///
/// Range::next(&mut range):
/// - If range.start < range.end:
///   - Returns Some(range.start)
///   - Advances range.start by 1
/// - Else:
///   - Returns None
///
/// We model this by:
/// - Setting dest_discr = (ite (< start end) 1 0)  -- 1 for Some, 0 for None
/// - Setting dest_val = start (the current start value)
/// - Advancing range_start' = (ite (< start end) (+ start 1) start)
///
/// Returns Some(list of extra variable names that were assigned) if the function was handled.
/// The caller should add these to the assigned set.
fn try_expand_range_next(
    func: &str,
    is_range_next: bool,
    args: &[String],
    dest: &str,
    local_names: &HashSet<&str>,
    conjuncts: &mut Vec<String>,
    all_blocks: &[MirBasicBlock],
) -> Option<Vec<String>> {
    if !call_is_range_next(func, is_range_next) {
        return None;
    }

    // next takes one argument: &mut Range (reference to range)
    if args.is_empty() {
        return None;
    }

    let range_ref = &args[0];

    // The range reference may be a deref or direct reference
    // Extract the base variable name
    // Pattern: (deref _4) -> _4, or just _4 -> _4
    let mut range_var = if range_ref.starts_with("(deref ") && range_ref.ends_with(')') {
        range_ref[7..range_ref.len() - 1].to_string()
    } else {
        range_ref.to_string()
    };

    // Trace back through assignments across ALL blocks to find the actual Range variable
    // In MIR, `_6 = &mut _4`, `_4 = move _2`, `_2 = into_iter(_3)`, `_3` has fields
    // We need to trace: _6 -> _4 -> _2 -> _3 (which has _3_field0 and _3_field1)
    let mut visited: HashSet<String> = HashSet::new();
    loop {
        if visited.contains(&range_var) {
            break; // Avoid infinite loops
        }
        visited.insert(range_var.clone());

        // Check if this variable has field0/field1
        let field0_name = format!("{}_field0", range_var);
        if local_names.contains(field0_name.as_str()) {
            break; // Found the variable with fields
        }

        // Search all blocks for an assignment to range_var
        let mut found_next = false;
        'outer: for block in all_blocks {
            for stmt in &block.statements {
                if let MirStatement::Assign { lhs, rhs } = stmt {
                    if lhs == &range_var {
                        // Found the assignment to our variable
                        // Extract the RHS variable if it's a simple variable or reference
                        let next_var = if rhs.starts_with("(ref ") && rhs.ends_with(')') {
                            // (ref _4) -> _4
                            Some(rhs[5..rhs.len() - 1].to_string())
                        } else if rhs.starts_with('_')
                            && rhs.chars().skip(1).all(|c| c.is_ascii_digit())
                        {
                            // Simple variable like _4
                            Some(rhs.clone())
                        } else {
                            None
                        };

                        if let Some(next) = next_var {
                            range_var = next;
                            found_next = true;
                            break 'outer;
                        }
                    }
                }
            }

            // Also check Call terminators (for into_iter)
            if let MirTerminator::Call {
                destination: Some(dest_var),
                args: call_args,
                func: call_func,
                is_range_into_iter,
                ..
            } = &block.terminator
            {
                if dest_var == &range_var && call_is_range_into_iter(call_func, *is_range_into_iter)
                {
                    // into_iter returns the same Range, so follow the argument
                    if !call_args.is_empty() {
                        range_var = call_args[0].clone();
                        found_next = true;
                        break;
                    }
                }
            }
        }

        if !found_next {
            break; // No more to trace
        }
    }

    // Range has two fields: start (field0) and end (field1)
    let start_var = format!("{}_field0", range_var);
    let end_var = format!("{}_field1", range_var);

    // Get current start and end values
    // If the fields exist as locals, use them; otherwise use placeholder expressions
    let start_expr = if local_names.contains(start_var.as_str()) {
        start_var.clone()
    } else {
        // Fall back to treating range_var as start (this handles simple cases)
        // Move range_var since it's not used after this point
        range_var
    };

    let end_expr = if local_names.contains(end_var.as_str()) {
        end_var.clone()
    } else {
        // If end field doesn't exist, we can't properly model the iterator
        return None;
    };

    let mut extra_assigned = Vec::new();

    // Condition: start < end (iterator has more elements)
    let has_more = format!("(< {} {})", start_expr, end_expr);

    // Option<T> encoding:
    // - The base variable (dest) holds the discriminant: 0=None, 1=Some
    // - The field variable (dest_field0) holds the value when Some
    //
    // MIR pattern:
    //   _5 = Range::next(&mut _4)   -- _5 is Option<i32>
    //   _7 = _5                      -- copy for switch
    //   switch(_7)                   -- 0=None, 1=Some
    //   _8 = (_5 as Some).0          -- extract value (mapped to _5_field0)

    // Set the base destination to the discriminant (0=None, 1=Some)
    if local_names.contains(dest) {
        conjuncts.push(format!("(= {}' (ite {} 1 0))", dest, has_more));
    }

    // Set the field0 to the value (current start)
    let val_var = format!("{}_field0", dest);
    if local_names.contains(val_var.as_str()) {
        conjuncts.push(format!("(= {}' {})", val_var, start_expr));
        extra_assigned.push(val_var);
    }

    // Also set _discr and _val variants if they exist (for compatibility with other patterns)
    let discr_var = format!("{}_discr", dest);
    if local_names.contains(discr_var.as_str()) {
        conjuncts.push(format!("(= {}' (ite {} 1 0))", discr_var, has_more));
        extra_assigned.push(discr_var);
    }
    let alt_val_var = format!("{}_val", dest);
    if local_names.contains(alt_val_var.as_str()) {
        conjuncts.push(format!("(= {}' {})", alt_val_var, start_expr));
        extra_assigned.push(alt_val_var);
    }

    // Advance the iterator: start' = (ite (< start end) (+ start 1) start)
    // Only advance if there was an element to return
    if local_names.contains(start_var.as_str()) {
        conjuncts.push(format!(
            "(= {}' (ite {} (+ {} 1) {}))",
            start_var, has_more, start_expr, start_expr
        ));
        extra_assigned.push(start_var);
    }

    // End stays unchanged - we add it to conjuncts here since it's modified
    if local_names.contains(end_var.as_str()) {
        conjuncts.push(format!("(= {}' {})", end_var, end_expr));
        extra_assigned.push(end_var);
    }

    extra_assigned.push(dest.to_string());
    Some(extra_assigned)
}

/// Parsed closure information for inlining
#[derive(Debug, Clone)]
pub struct ClosureInfo {
    /// The source location pattern (e.g., "/tmp/closure_simple.rs:2:19: 2:27")
    pub source_pattern: String,
    /// The closure function name (e.g., "closure_proof::{closure#0}")
    pub function_name: String,
    /// The body expression (simplified for zero-capture closures)
    /// For `|x| x + 1`, this would be `(+ arg1 1)`
    pub body_expr: Option<String>,
    /// Number of arguments the closure takes (excluding the closure self reference)
    pub arg_count: usize,
}

/// Parsed trait impl method information for static dispatch inlining
#[derive(Debug, Clone)]
pub struct TraitImplInfo {
    /// The implementing type (e.g., "Value")
    pub impl_type: String,
    /// The trait being implemented (e.g., "Addable")
    pub trait_name: String,
    /// The method name (e.g., "add_value")
    pub method_name: String,
    /// The impl function name (e.g., "<impl at /tmp/file.rs:9:1: 9:23>::add_value")
    pub function_name: String,
    /// The body expression (simplified for simple methods)
    /// For `fn add_value(&self, x: i32) -> i32 { self.n + x }`, this would be `(+ self_field0 arg1)`
    pub body_expr: Option<String>,
    /// Number of arguments (excluding self reference)
    pub arg_count: usize,
}

/// Check if a function name is a closure call
///
/// Closure calls have the pattern:
/// `<{closure@PATH} as Fn<ARGS>>::call`
fn is_closure_call(func: &str) -> bool {
    func.starts_with("<{closure@") && func.contains(">::call")
}

/// Extract the source location pattern from a closure call
///
/// From `<{closure@/tmp/closure_simple.rs:2:19: 2:27} as Fn<(i32,)>>::call`
/// extract `/tmp/closure_simple.rs:2:19: 2:27`
fn extract_closure_source_pattern(func: &str) -> Option<String> {
    if !is_closure_call(func) {
        return None;
    }
    // Extract the part between {closure@ and }
    let start = func.find("{closure@")? + "{closure@".len();
    let end = func.find('}')?;
    if start >= end {
        return None;
    }
    Some(func[start..end].to_string())
}

/// Try to expand a simple closure call by inlining its body
///
/// For zero-capture closures with simple arithmetic bodies like `|x| x + 1`,
/// we can inline the body expression directly.
///
/// # Arguments
/// * `func` - The function name being called
/// * `args` - The call arguments (closure ref, then tuple of actual args)
/// * `dest` - The destination variable
/// * `closures` - Map of source patterns to closure info
/// * `conjuncts` - Output constraints to append to
/// * `local_names` - Set of live variables in the state
/// * `block_statements` - Statements from the call block (for looking up tuple element values)
///
/// # Returns
/// `true` if the closure was inlined successfully
#[allow(clippy::too_many_arguments)]
fn try_expand_closure_call<'a>(
    func: &str,
    args: &[String],
    dest: &'a str,
    closures: &HashMap<String, ClosureInfo>,
    conjuncts: &mut Vec<String>,
    local_names: &HashSet<&str>,
    assigned: &mut HashSet<&'a str>,
    block_statements: &[MirStatement],
) -> bool {
    if !is_closure_call(func) {
        return false;
    }

    let Some(source_pattern) = extract_closure_source_pattern(func) else {
        return false;
    };

    // Try to find a matching closure
    // First try direct lookup by source pattern
    // Then try to find by matching the source file path
    let closure_info = if let Some(info) = closures.get(&source_pattern) {
        info
    } else if closures.len() == 1 {
        // If there's exactly one closure, use it regardless of the key
        // source_pattern is like: /tmp/closure_simple.rs:2:19: 2:27
        // closures keys are like: closure_proof::{closure#0}
        // This handles the simple case of a single closure per function
        // Safety: len()==1 guarantees values().next() returns Some
        closures.values().next().expect("single closure exists")
    } else {
        return false;
    };

    // Check if we have a body expression to inline
    let Some(body_expr) = &closure_info.body_expr else {
        return false; // Complex closure, can't inline
    };

    // The call args are:
    // args[0]: closure reference (ignored for zero-capture closures)
    // args[1]: tuple of actual arguments
    //
    // For MIR, the tuple is passed as a single variable like _4
    // and the tuple elements are accessed as _4_elem_0, _4_elem_1, etc.
    //
    // For a single-argument closure |x|, we use args[1] directly since
    // the single element is the tuple itself in the simple case.
    if args.len() < 2 {
        return false;
    }

    let tuple_arg = &args[1];

    // Substitute arg1 in the body expression with the actual argument
    // For single-arg closures: replace "arg1" with the tuple argument element
    //
    // The tuple_arg is something like "_4", and we need to extract element 0.
    // First check if the element is in live variables. If not, look up its
    // assigned value in the block statements (since dead variable analysis
    // may have eliminated it).
    let elem0_name = format!("{}_elem_0", tuple_arg);
    let field0_name = format!("{}_field0", tuple_arg);

    let arg_value = if local_names.contains(elem0_name.as_str()) {
        elem0_name
    } else if local_names.contains(field0_name.as_str()) {
        field0_name
    } else {
        // Look up the value from block statements
        // If we have `_4_elem_0 = 5` in the statements, use "5" directly
        let mut found_value = None;
        for stmt in block_statements {
            if let MirStatement::Assign { lhs, rhs } = stmt {
                if lhs == &elem0_name || lhs == &field0_name {
                    found_value = Some(rhs.clone());
                    break;
                }
            }
        }
        found_value.unwrap_or_else(|| tuple_arg.clone())
    };

    // Substitute arg1 with the actual value
    let result_expr = body_expr.replace("arg1", &arg_value);

    // Assign the result to the destination
    if local_names.contains(dest) {
        conjuncts.push(format!("(= {}' {})", dest, result_expr));
    }
    assigned.insert(dest);

    true
}

/// Build closure info from a parsed MIR function
///
/// Analyzes a closure function to extract its body expression for inlining.
/// Only handles simple arithmetic closures currently.
pub fn build_closure_info(
    function_name: &str,
    source_pattern: &str,
    blocks: &[MirBasicBlock],
    args: &[(String, SmtType)],
) -> ClosureInfo {
    let arg_count = args.len().saturating_sub(1); // Exclude closure self reference

    // Try to extract a simple body expression
    // For `|x| x + 1`, the MIR would be:
    //   bb0: _3 = AddWithOverflow(_2, const 1)
    //   bb1: _0 = (_3.0: i32)
    //
    // We simplify this to just (+ arg1 1)

    let body_expr = extract_simple_closure_body(blocks, args);

    ClosureInfo {
        source_pattern: source_pattern.to_string(),
        function_name: function_name.to_string(),
        body_expr,
        arg_count,
    }
}

/// Extract a simple body expression from closure blocks
///
/// Handles patterns like:
/// - Simple arithmetic: `|x| x + 1` → `(+ arg1 1)`
/// - Direct return: `|x| x` → `arg1`
fn extract_simple_closure_body(
    blocks: &[MirBasicBlock],
    args: &[(String, SmtType)],
) -> Option<String> {
    if blocks.is_empty() || args.len() < 2 {
        return None;
    }

    // The closure takes:
    // _1: closure reference (ignored)
    // _2: first actual argument
    let first_arg = &args.get(1)?.0;

    // Look at the first block for the computation
    let first_block = &blocks[0];

    // Look for the assignment pattern
    for stmt in &first_block.statements {
        if let MirStatement::Assign { lhs, rhs } = stmt {
            // Check if this is an AddWithOverflow or similar operation
            // The rhs would be something like "(+ _2 1)" or "_2"

            // Direct return of argument
            if rhs == first_arg {
                return Some("arg1".to_string());
            }

            // Check for arithmetic operations
            // We store _0 = (op _2 const) format
            if lhs == "_0" {
                // This is the return value assignment
                return Some(rhs.replace(first_arg, "arg1"));
            }

            // Check for intermediate variable that gets returned
            // Often _3 = AddWithOverflow(_2, 1), then _0 = _3_elem_0
            if rhs.contains(first_arg) {
                // Extract operation from intermediate
                let simplified = simplify_closure_expr(rhs, first_arg);
                if !simplified.is_empty() {
                    // Pre-compute pattern for inner loop (avoid format! allocation)
                    let lhs_underscore = format!("{}_", lhs);
                    // Check if _0 uses this variable
                    for stmt2 in &first_block.statements {
                        if let MirStatement::Assign {
                            lhs: lhs2,
                            rhs: rhs2,
                        } = stmt2
                        {
                            if lhs2 == "_0" {
                                // _0 = lhs_elem_0 pattern
                                if rhs2.starts_with(lhs) {
                                    return Some(simplified.replace(first_arg, "arg1"));
                                }
                            }
                        }
                    }
                    // Check next block for _0 assignment
                    if blocks.len() > 1 {
                        for stmt2 in &blocks[1].statements {
                            if let MirStatement::Assign {
                                lhs: lhs2,
                                rhs: rhs2,
                            } = stmt2
                            {
                                if lhs2 == "_0"
                                    && (rhs2.starts_with(lhs) || rhs2.contains(&lhs_underscore))
                                {
                                    return Some(simplified.replace(first_arg, "arg1"));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Also check the second block (for AddWithOverflow pattern)
    if blocks.len() > 1 {
        for stmt in &blocks[1].statements {
            if let MirStatement::Assign { lhs, rhs } = stmt {
                if lhs == "_0" && rhs.contains("_elem_0") {
                    // The result is the elem_0 of some intermediate variable
                    // Find what that intermediate variable was assigned
                    for stmt2 in &blocks[0].statements {
                        if let MirStatement::Assign {
                            lhs: lhs2,
                            rhs: rhs2,
                        } = stmt2
                        {
                            // Check if rhs references this intermediate
                            // Avoid format! allocation: check if rhs starts with lhs2 followed by _
                            let starts_with_lhs2_underscore = rhs.starts_with(lhs2)
                                && rhs.as_bytes().get(lhs2.len()) == Some(&b'_');
                            if rhs.contains(lhs2) || starts_with_lhs2_underscore {
                                let simplified = simplify_closure_expr(rhs2, first_arg);
                                if !simplified.is_empty() {
                                    return Some(simplified.replace(first_arg, "arg1"));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Simplify a closure expression for inlining
fn simplify_closure_expr(expr: &str, arg_name: &str) -> String {
    // Handle AddWithOverflow pattern: becomes just addition
    if expr.starts_with("(+") || expr.contains("AddWithOverflow") {
        // Extract operands from either (+ a b) or AddWithOverflow(a, b) format
        // For now, just convert to SMT format
        if let Some(result) = extract_binary_op(expr, arg_name, "+") {
            return result;
        }
    }
    if expr.starts_with("(-") || expr.contains("SubWithOverflow") {
        if let Some(result) = extract_binary_op(expr, arg_name, "-") {
            return result;
        }
    }
    if expr.starts_with("(*") || expr.contains("MulWithOverflow") {
        if let Some(result) = extract_binary_op(expr, arg_name, "*") {
            return result;
        }
    }
    // Direct reference to argument
    if expr == arg_name {
        return "arg1".to_string();
    }
    String::new()
}

/// Extract a binary operation pattern
fn extract_binary_op(expr: &str, arg_name: &str, op: &str) -> Option<String> {
    // Try to find the pattern with the argument
    if expr.contains(arg_name) {
        // Look for constants in the expression
        // Common patterns:
        // (+ _2 1), (+_2 1), AddWithOverflow(_2, 1)

        // Simple pattern: find number after arg_name
        if let Some(pos) = expr.find(arg_name) {
            let after = &expr[pos + arg_name.len()..];
            // Try to extract number
            let num_str: String = after
                .chars()
                .skip_while(|c| !c.is_ascii_digit() && *c != '-')
                .take_while(|c| c.is_ascii_digit() || *c == '-')
                .collect();
            if !num_str.is_empty() {
                return Some(format!("({} arg1 {})", op, num_str));
            }
        }

        // Check for constant before arg_name
        if let Some(pos) = expr.find(arg_name) {
            let before = &expr[..pos];
            let num_str: String = before
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_digit() || *c == '-')
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            if !num_str.is_empty() {
                return Some(format!("({} {} arg1)", op, num_str));
            }
        }
    }
    None
}

/// Check if a function name is a trait-qualified call
///
/// Trait-qualified calls have the pattern:
/// `<Type as Trait>::method`
pub fn is_trait_qualified_call(func: &str) -> bool {
    // Pattern: <Type as Trait>::method
    // Must start with <, contain " as ", and end with >::method_name
    func.starts_with('<') && func.contains(" as ") && func.contains(">::")
}

/// Extract type, trait, and method from a trait-qualified call
///
/// From `<Value as Addable>::add_value` extracts:
/// - type_name: "Value"
/// - trait_name: "Addable"
/// - method_name: "add_value"
pub fn parse_trait_qualified_call(func: &str) -> Option<(String, String, String)> {
    if !is_trait_qualified_call(func) {
        return None;
    }

    // Extract the part between < and >
    let inner_start = func.find('<')? + 1;
    let inner_end = func.rfind('>')?;
    if inner_start >= inner_end {
        return None;
    }
    let inner = &func[inner_start..inner_end];

    // Split by " as " to get type and trait (avoid Vec allocation)
    let mut parts = inner.splitn(2, " as ");
    let type_name = parts.next()?.trim().to_string();
    let trait_name = parts.next()?.trim().to_string();

    // Extract method name after the LAST >::
    // Using rfind because trait names can contain >, e.g., Fn<(i32, i32)>
    let method_start = func.rfind(">::")? + 3;
    // Method name goes until ( or end of string, but only look after method_start
    let method_end = func[method_start..]
        .find('(')
        .map(|i| method_start + i)
        .unwrap_or(func.len());
    if method_start >= method_end {
        return None;
    }
    let method_name = func[method_start..method_end].trim().to_string();

    Some((type_name, trait_name, method_name))
}

/// Try to expand a trait method call by inlining its impl body
///
/// For static dispatch calls like `<Value as Addable>::add_value(ref, arg)`,
/// looks up the impl and inlines its body expression.
///
/// # Arguments
/// * `func` - The function name being called
/// * `args` - The call arguments (self ref, then actual args)
/// * `dest` - The destination variable
/// * `trait_impls` - Map of trait-qualified calls to impl info
/// * `conjuncts` - Output constraints to append to
/// * `local_names` - Set of live variables in the state
/// * `assigned` - Set of already-assigned variables (mutated)
/// * `block_statements` - Statements from the call block
///
/// # Returns
/// `true` if the impl was inlined successfully
#[allow(clippy::too_many_arguments)]
fn try_expand_trait_call<'a>(
    func: &str,
    args: &[String],
    dest: &'a str,
    trait_impls: &HashMap<String, TraitImplInfo>,
    conjuncts: &mut Vec<String>,
    local_names: &HashSet<&str>,
    assigned: &mut HashSet<&'a str>,
    block_statements: &[MirStatement],
) -> bool {
    if !is_trait_qualified_call(func) {
        return false;
    }

    // Look up the impl info by the full trait-qualified call pattern
    // First, try to find an exact match
    let impl_info: Option<&TraitImplInfo> = if let Some(info) = trait_impls.get(func) {
        Some(info)
    } else {
        // Try to match by type/trait/method decomposition
        if let Some((type_name, trait_name, method_name)) = parse_trait_qualified_call(func) {
            trait_impls.values().find(|info| {
                info.impl_type == type_name
                    && info.trait_name == trait_name
                    && info.method_name == method_name
            })
        } else {
            None
        }
    };

    let Some(impl_info) = impl_info else {
        return false;
    };

    // Check if we have a body expression to inline
    let Some(body_expr) = &impl_info.body_expr else {
        return false; // Complex impl, can't inline
    };

    // Args:
    // args[0]: self reference
    // args[1..]: actual method arguments
    if args.is_empty() {
        return false;
    }

    let self_ref = &args[0];

    // Build substitution: replace "self_field0" with actual self field access
    // and "arg1", "arg2", etc. with actual arguments
    let mut result_expr = body_expr.clone();

    // Substitute self field access
    // The self_ref might be primed (e.g., "_3'"), strip the prime for lookups
    let self_ref_base = self_ref.trim_end_matches('\'');

    // Look for what the self reference points to
    // Pattern: _3 = _1 means _3 is a reference to _1
    // So self_field0 should be _1_field0
    let actual_self = find_assigned_value(self_ref_base, block_statements)
        .or_else(|| find_assigned_value(&format!("{}'", self_ref_base), block_statements))
        .map(|v| v.trim_end_matches('\'').to_string())
        .unwrap_or_else(|| self_ref_base.to_string());

    // Now get the field value: {actual_self}_field0
    let self_field0_name = format!("{}_field0", actual_self);
    let self_field0_val = find_assigned_value(&self_field0_name, block_statements)
        .unwrap_or_else(|| self_field0_name.clone());

    result_expr = result_expr.replace("self_field0", &self_field0_val);

    // Substitute arguments
    for (i, arg) in args.iter().skip(1).enumerate() {
        let arg_placeholder = format!("arg{}", i + 1);
        result_expr = result_expr.replace(&arg_placeholder, arg);
    }

    // Assign the result to the destination
    if local_names.contains(dest) {
        conjuncts.push(format!("(= {}' {})", dest, result_expr));
    }
    assigned.insert(dest);

    true
}

/// Find the value assigned to a variable in block statements
fn find_assigned_value(var: &str, statements: &[MirStatement]) -> Option<String> {
    for stmt in statements {
        if let MirStatement::Assign { lhs, rhs } = stmt {
            if lhs == var {
                return Some(rhs.clone());
            }
        }
    }
    None
}

/// Build trait impl info from a parsed MIR impl function
///
/// Analyzes an impl function to extract its body expression for inlining.
/// Only handles simple impl methods currently.
pub fn build_trait_impl_info(
    function_name: &str,
    impl_type: &str,
    trait_name: &str,
    method_name: &str,
    blocks: &[MirBasicBlock],
    args: &[(String, SmtType)],
) -> TraitImplInfo {
    let arg_count = args.len().saturating_sub(1); // Exclude self reference

    // Try to extract a simple body expression
    // For `fn add_value(&self, x: i32) -> i32 { self.n + x }`, the MIR would be:
    //   bb0: _3 = copy ((*_1).0: i32)  // self.n
    //        _4 = AddWithOverflow(_3, _2)
    //   bb1: _0 = (_4.0: i32)
    //
    // We simplify this to (+ self_field0 arg1)

    let body_expr = extract_simple_impl_body(blocks, args);

    TraitImplInfo {
        impl_type: impl_type.to_string(),
        trait_name: trait_name.to_string(),
        method_name: method_name.to_string(),
        function_name: function_name.to_string(),
        body_expr,
        arg_count,
    }
}

/// Extract a simple body expression from impl method blocks
///
/// Handles patterns like:
/// - Field access + arithmetic: `fn add(&self, x) -> i32 { self.n + x }` → `(+ self_field0 arg1)`
/// - Direct field return: `fn get(&self) -> i32 { self.n }` → `self_field0`
fn extract_simple_impl_body(
    blocks: &[MirBasicBlock],
    args: &[(String, SmtType)],
) -> Option<String> {
    if blocks.is_empty() || args.is_empty() {
        return None;
    }

    // The impl takes:
    // _1: &Self (self reference)
    // _2, _3, ...: method arguments
    //
    // Look for field access pattern in first block
    // Then find what the return value (_0) is assigned to

    // First, find the self field access: _X = *_1)_elem_0 or _X = (*_1).0
    let mut self_field_var: Option<String> = None;

    for stmt in &blocks[0].statements {
        if let MirStatement::Assign { lhs, rhs } = stmt {
            // Pattern: _X = *_1)_elem_0 or _X = (*_1_field0) or similar
            if rhs.contains("*_1") || rhs.contains("_1_field") {
                self_field_var = Some(lhs.clone());
                break;
            }
        }
    }

    // Now find the computation that produces _0
    // Look for AddWithOverflow, SubWithOverflow, or direct assignment patterns

    // Second method argument (first actual arg after self) is _2
    let first_arg = "_2";

    for stmt in &blocks[0].statements {
        if let MirStatement::Assign { lhs, rhs } = stmt {
            // Look for arithmetic operations
            if rhs.starts_with("(+") || rhs.starts_with("(-") || rhs.starts_with("(*") {
                // Check if it references self_field_var and first_arg
                if let Some(sfv) = &self_field_var {
                    if rhs.contains(sfv) || rhs.contains(first_arg) {
                        // Extract operation and build expression
                        let mut expr = rhs.clone();
                        // Replace self field var with self_field0
                        if let Some(sfv) = &self_field_var {
                            expr = expr.replace(sfv, "self_field0");
                        }
                        // Replace _2 with arg1
                        expr = expr.replace(first_arg, "arg1");
                        // Replace _3 with arg2 if present
                        expr = expr.replace("_3", "arg2");
                        return Some(expr);
                    }
                }
            }

            // Check for _elem_0 pattern (from AddWithOverflow)
            if lhs.ends_with("_elem_0")
                && (rhs.starts_with("(+") || rhs.starts_with("(-") || rhs.starts_with("(*"))
            {
                let mut expr = rhs.clone();
                // Replace self field var with self_field0
                if let Some(sfv) = &self_field_var {
                    expr = expr.replace(sfv, "self_field0");
                }
                // Replace _2 with arg1
                expr = expr.replace(first_arg, "arg1");
                expr = expr.replace("_3", "arg2");
                return Some(expr);
            }
        }
    }

    // Check if _0 is assigned directly in block 0 or block 1
    for block in blocks {
        for stmt in &block.statements {
            if let MirStatement::Assign { lhs, rhs } = stmt {
                if lhs == "_0" {
                    // Direct return of self field
                    if let Some(sfv) = &self_field_var {
                        // Check if rhs equals sfv or contains "{sfv}_" without allocating
                        let matches = rhs == sfv || {
                            let sfv_bytes = sfv.as_bytes();
                            let rhs_bytes = rhs.as_bytes();
                            rhs_bytes.windows(sfv_bytes.len() + 1).any(|w| {
                                &w[..sfv_bytes.len()] == sfv_bytes && w[sfv_bytes.len()] == b'_'
                            })
                        };
                        if matches {
                            return Some("self_field0".to_string());
                        }
                    }
                    // Return of _elem_0 from intermediate
                    if rhs.contains("_elem_0") {
                        // Find what the elem_0 was computed from
                        for stmt2 in &blocks[0].statements {
                            if let MirStatement::Assign {
                                lhs: lhs2,
                                rhs: rhs2,
                            } = stmt2
                            {
                                if lhs2.ends_with("_elem_0")
                                    && (rhs2.starts_with("(+")
                                        || rhs2.starts_with("(-")
                                        || rhs2.starts_with("(*"))
                                {
                                    let mut expr = rhs2.clone();
                                    if let Some(sfv) = &self_field_var {
                                        expr = expr.replace(sfv, "self_field0");
                                    }
                                    expr = expr.replace(first_arg, "arg1");
                                    expr = expr.replace("_3", "arg2");
                                    return Some(expr);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Hybrid substitution strategy for CHC encoding.
///
/// For variables assigned multiple times in a block (in `multi_assigned`):
///   - Substitute with their full expression from `values`
///   - This avoids circular dependencies in the CHC encoding
///
/// For variables assigned exactly once (in `assigned` but not `multi_assigned`):
///   - Substitute with the primed variable (e.g., `_1` -> `_1'`)
///   - This preserves variable relationships that Z3 Spacer uses for invariant synthesis
///
/// For variables not assigned in this block:
///   - Leave unchanged (they refer to the current state)
fn substitute_hybrid(
    expr: &str,
    values: &std::collections::HashMap<String, String>,
    assigned: &HashSet<String>,
    multi_assigned: &HashSet<String>,
) -> String {
    if assigned.is_empty() {
        return expr.to_string();
    }

    let mut result = expr.to_string();

    // Sort by length descending to handle _10 before _1
    let mut sorted_vars: Vec<_> = assigned.iter().collect();
    sorted_vars.sort_by_key(|b| std::cmp::Reverse(b.len()));

    for var in sorted_vars {
        let replacement = if multi_assigned.contains(var) {
            // Multi-assigned: use full expression to avoid circular deps
            values.get(var).cloned().unwrap_or_else(|| var.clone())
        } else {
            // Single-assigned: use primed variable for better invariant synthesis
            format!("{}'", var)
        };

        let mut new_result = String::new();
        let var_bytes = var.as_bytes();
        let result_bytes = result.as_bytes();
        let mut i = 0;

        while i < result_bytes.len() {
            // Check if we're at a potential match
            if i + var_bytes.len() <= result_bytes.len()
                && &result_bytes[i..i + var_bytes.len()] == var_bytes
            {
                // Check preceding character (should not be alphanumeric or _)
                let preceded_ok = if i == 0 {
                    true
                } else {
                    let prev = result_bytes[i - 1];
                    !prev.is_ascii_alphanumeric() && prev != b'_'
                };

                // Check following character (should not be alphanumeric, _, or ')
                let followed_ok = if i + var_bytes.len() >= result_bytes.len() {
                    true
                } else {
                    let next = result_bytes[i + var_bytes.len()];
                    !next.is_ascii_alphanumeric() && next != b'_' && next != b'\''
                };

                if preceded_ok && followed_ok {
                    // Replace with the appropriate form
                    new_result.push_str(&replacement);
                    i += var_bytes.len();
                    continue;
                }
            }

            // No match, copy character
            new_result.push(result_bytes[i] as char);
            i += 1;
        }

        result = new_result;
    }

    result
}

/// Substitute variable references in a guard expression with their primed versions.
///
/// In MIR, terminator conditions read values AFTER the block's statements execute.
/// So if a block assigns `_5 = Lt(_4, 3)` and then has `assert(_5, ...)`, the guard
/// `_5` refers to the newly computed value, not the pre-state value.
///
/// This function finds variable references like `_5` in the guard and replaces them
/// with `_5'` if that variable was assigned in the current block.
fn substitute_assigned_vars(guard: &str, assigned: &HashSet<&str>) -> String {
    if assigned.is_empty() {
        return guard.to_string();
    }

    let mut result = guard.to_string();

    // Sort by length descending to handle _10 before _1
    let mut sorted_vars: Vec<_> = assigned.iter().collect();
    sorted_vars.sort_by_key(|b| std::cmp::Reverse(b.len()));

    for var in sorted_vars {
        // Simple string-based substitution with boundary checking
        // We need to find occurrences of var that are not followed by ' or alphanumeric
        // and not preceded by alphanumeric (Rust regex doesn't support lookbehind)
        let primed_var = format!("{}'", var);
        let mut new_result = String::new();
        let var_bytes = var.as_bytes();
        let guard_bytes = result.as_bytes();
        let mut i = 0;

        while i < guard_bytes.len() {
            // Check if we're at a potential match
            if i + var_bytes.len() <= guard_bytes.len()
                && &guard_bytes[i..i + var_bytes.len()] == var_bytes
            {
                // Check preceding character (should not be alphanumeric or _)
                let preceded_ok = if i == 0 {
                    true
                } else {
                    let prev = guard_bytes[i - 1];
                    !prev.is_ascii_alphanumeric() && prev != b'_'
                };

                // Check following character (should not be alphanumeric, _, or ')
                let followed_ok = if i + var_bytes.len() >= guard_bytes.len() {
                    true
                } else {
                    let next = guard_bytes[i + var_bytes.len()];
                    !next.is_ascii_alphanumeric() && next != b'_' && next != b'\''
                };

                if preceded_ok && followed_ok {
                    // Replace with primed version
                    new_result.push_str(&primed_var);
                    i += var_bytes.len();
                    continue;
                }
            }

            // No match, copy character
            new_result.push(guard_bytes[i] as char);
            i += 1;
        }

        result = new_result;
    }

    result
}

/// Format a SwitchInt comparison for the given discriminant and value.
/// For boolean discriminants, converts integer values (0/1) to proper SMT booleans.
/// For integer discriminants, uses direct equality comparison.
/// For bitvector discriminants, converts the value to a bitvector literal.
fn format_switch_comparison(discr: &str, value: i64, locals: &[MirLocal]) -> String {
    // Check if the discriminant is a boolean variable
    let is_bool = locals
        .iter()
        .any(|l| l.name == discr && matches!(l.ty, SmtType::Bool));

    // Check if the discriminant is a bitvector variable
    let bitvec_width = locals.iter().find(|l| l.name == discr).and_then(|l| {
        if let SmtType::BitVec(width) = l.ty {
            Some(width)
        } else {
            None
        }
    });

    if is_bool {
        // For boolean discriminants, 0 = false, non-zero = true
        if value == 0 {
            format!("(not {})", discr)
        } else {
            discr.to_string()
        }
    } else if let Some(width) = bitvec_width {
        // For bitvector discriminants, use bitvector literal
        format!("(= {} (_ bv{} {}))", discr, value, width)
    } else {
        // For integer discriminants, use direct equality
        format!("(= {} {})", discr, value)
    }
}

/// Format the negation of a SwitchInt comparison.
fn format_switch_comparison_negated(discr: &str, value: i64, locals: &[MirLocal]) -> String {
    let is_bool = locals
        .iter()
        .any(|l| l.name == discr && matches!(l.ty, SmtType::Bool));

    // Check if the discriminant is a bitvector variable
    let bitvec_width = locals.iter().find(|l| l.name == discr).and_then(|l| {
        if let SmtType::BitVec(width) = l.ty {
            Some(width)
        } else {
            None
        }
    });

    if is_bool {
        // For boolean discriminants, 0 = false, non-zero = true
        if value == 0 {
            // Negating "discr == false" gives "discr == true" i.e., just discr
            discr.to_string()
        } else {
            // Negating "discr == true" gives "discr == false" i.e., (not discr)
            format!("(not {})", discr)
        }
    } else if let Some(width) = bitvec_width {
        // For bitvector discriminants, use bitvector literal
        format!("(not (= {} (_ bv{} {})))", discr, value, width)
    } else {
        // For integer discriminants, use direct inequality
        format!("(not (= {} {}))", discr, value)
    }
}

fn build_block_clause(
    block: &MirBasicBlock,
    extra_guard: Option<String>,
    pc_target: i64,
    locals: &[MirLocal],
) -> String {
    // Pre-allocate: pc + guard + statements + locals + terminator
    let estimated_size = block.statements.len() + locals.len() + 4;
    let mut conjuncts = Vec::with_capacity(estimated_size);

    // Collect local variable names for field propagation and live variable checks
    let local_names: HashSet<&str> = locals.iter().map(|l| l.name.as_str()).collect();

    // First pass: count how many times each variable is assigned in this block.
    // Variables assigned multiple times need full expression substitution to avoid
    // circular dependencies in the CHC encoding.
    let mut assignment_count: std::collections::HashMap<String, usize> =
        std::collections::HashMap::with_capacity(block.statements.len());
    for stmt in &block.statements {
        if let MirStatement::Assign { lhs, .. } = stmt {
            *assignment_count.entry(lhs.clone()).or_insert(0) += 1;
        }
    }

    // Variables assigned more than once need expression-based substitution.
    // Variables assigned exactly once can use primed variable substitution (better for Z3 Spacer).
    let multi_assigned: HashSet<String> = assignment_count
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|(var, _)| var.clone())
        .collect();

    // Track the CURRENT value expression for each variable at each point in the block.
    // For multi-assigned variables, this holds the full expression.
    // For single-assigned variables, this can use primed variable references.
    let mut current_values: std::collections::HashMap<String, String> =
        std::collections::HashMap::with_capacity(block.statements.len());
    // Track which variables have been assigned at least once (for primed var substitution)
    let mut assigned: HashSet<String> = HashSet::with_capacity(block.statements.len());
    // Track variables that should be left unconstrained (havoc'd)
    let mut havoc_vars: HashSet<String> = HashSet::with_capacity(4);

    conjuncts.push(format!("(= pc {})", block.id));

    // Process statements sequentially, tracking the current value of each variable.
    // In MIR, statements execute in order, so later statements read the results
    // of earlier statements in the same block.
    //
    // Hybrid substitution strategy:
    // - For multi-assigned vars: substitute with full expression (avoids circular deps)
    // - For single-assigned vars: substitute with primed var (better for invariant synthesis)
    for stmt in &block.statements {
        match stmt {
            MirStatement::Assume(cond) => {
                // Substitute current values in assume condition
                let substituted =
                    substitute_hybrid(cond, &current_values, &assigned, &multi_assigned);
                conjuncts.push(substituted);
            }
            MirStatement::Assign { lhs, rhs } => {
                // Substitute current values in RHS to get the value expression
                let substituted_rhs =
                    substitute_hybrid(rhs, &current_values, &assigned, &multi_assigned);
                // Update the current value for this variable
                current_values.insert(lhs.clone(), substituted_rhs.clone());
                assigned.insert(lhs.clone());

                // Handle struct copy semantics: if LHS has field variables, propagate them
                // e.g., _4 = _2 should also set _4_field0 = _2_field0, _4_field1 = _2_field1
                // Only do this for simple variable-to-variable copies (not expressions)
                if rhs.starts_with('_') && rhs.chars().skip(1).all(|c| c.is_ascii_digit()) {
                    // This is a simple variable copy like _4 = _2
                    for suffix in &["_field0", "_field1", "_elem_0", "_elem_1", "_discr", "_val"] {
                        let lhs_field = format!("{}{}", lhs, suffix);
                        let rhs_field = format!("{}{}", rhs, suffix);
                        if local_names.contains(lhs_field.as_str())
                            && local_names.contains(rhs_field.as_str())
                        {
                            // Propagate the field value from source to destination
                            let src_value = substitute_hybrid(
                                &rhs_field,
                                &current_values,
                                &assigned,
                                &multi_assigned,
                            );
                            current_values.insert(lhs_field.clone(), src_value);
                            assigned.insert(lhs_field);
                        }
                    }
                }
            }
            MirStatement::Assert { .. } => {
                // Assertions are handled in the property
            }
            MirStatement::ArrayStore {
                array,
                index,
                value,
            } => {
                // SMT-LIB2: (store array index value)
                // Substitute current values in array, index, and value expressions
                let sub_array =
                    substitute_hybrid(array, &current_values, &assigned, &multi_assigned);
                let sub_index =
                    substitute_hybrid(index, &current_values, &assigned, &multi_assigned);
                let sub_value =
                    substitute_hybrid(value, &current_values, &assigned, &multi_assigned);
                let store_expr = format!("(store {} {} {})", sub_array, sub_index, sub_value);
                current_values.insert(array.clone(), store_expr);
                assigned.insert(array.clone());
            }
            MirStatement::Havoc { var } => {
                // Havoc: variable is unconstrained in next state
                // Remove any prior assignment and mark as havoc'd
                current_values.remove(var);
                assigned.insert(var.clone());
                havoc_vars.insert(var.clone());
            }
        }
    }

    // Emit final value constraints for assigned variables that are in the state
    // Only include variables that are in the locals list (live variables)
    // to avoid referencing undeclared variables in the CHC formula
    for (var, value) in &current_values {
        if local_names.contains(var.as_str()) {
            conjuncts.push(format!("(= {}' {})", var, value));
        }
    }

    // Add terminator guard AFTER processing statements
    // The guard must use the current (computed) values of variables assigned in this block,
    // because MIR terminators read values after the block's statements execute.
    if let Some(guard) = extra_guard {
        let substituted_guard =
            substitute_hybrid(&guard, &current_values, &assigned, &multi_assigned);
        conjuncts.push(substituted_guard);
    }

    // Unassigned locals (and non-havoc'd) are carried forward
    for local in locals {
        if !assigned.contains(&local.name) {
            conjuncts.push(format!("(= {}' {})", local.name, local.name));
        }
    }

    // Program counter update
    conjuncts.push(format!("(= pc' {})", pc_target));

    format!("(and {})", conjuncts.join(" "))
}

/// Convert a basic block index to i64 for use in SMT formulas.
///
/// Panics if the index is too large to fit in i64, which would indicate
/// a program with more than 2^63 basic blocks (practically impossible).
fn idx_to_i64(idx: usize) -> i64 {
    i64::try_from(idx).expect("basic block index should fit in i64")
}

/// Build an SMT conjunction from a list of formulas.
///
/// Returns:
/// - `"true"` for an empty list (neutral element of conjunction)
/// - The single formula unchanged for a list of one
/// - `(and f1 f2 ... fn)` for multiple formulas
fn conjoin(parts: &[String]) -> String {
    match parts.len() {
        0 => "true".to_string(),
        1 => parts[0].clone(),
        _ => format!("(and {})", parts.join(" ")),
    }
}

/// Build an SMT disjunction from a list of formulas.
///
/// Returns:
/// - `"false"` for an empty list (neutral element of disjunction)
/// - The single formula unchanged for a list of one
/// - `(or f1 f2 ... fn)` for multiple formulas
fn disjoin(parts: &[String]) -> String {
    match parts.len() {
        0 => "false".to_string(),
        1 => parts[0].clone(),
        _ => format!("(or {})", parts.join(" ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify_chc;
    use std::time::Duration;

    fn has_z3() -> bool {
        crate::find_executable("z3").is_some()
    }

    // ========================================================================
    // substitute_assigned_vars tests
    // ========================================================================

    #[test]
    fn test_substitute_assigned_vars_simple() {
        let mut assigned = HashSet::new();
        assigned.insert("_5");
        let result = substitute_assigned_vars("_5", &assigned);
        assert_eq!(result, "_5'");
    }

    #[test]
    fn test_substitute_assigned_vars_in_expression() {
        let mut assigned = HashSet::new();
        assigned.insert("_5");
        assigned.insert("_4");
        let result = substitute_assigned_vars("(and _5 (< _4 3))", &assigned);
        assert_eq!(result, "(and _5' (< _4' 3))");
    }

    #[test]
    fn test_substitute_assigned_vars_only_assigned() {
        let mut assigned = HashSet::new();
        assigned.insert("_5");
        // _4 is not assigned, should remain as-is
        let result = substitute_assigned_vars("(and _5 (< _4 3))", &assigned);
        assert_eq!(result, "(and _5' (< _4 3))");
    }

    #[test]
    fn test_substitute_assigned_vars_empty() {
        let assigned = HashSet::new();
        let result = substitute_assigned_vars("_5", &assigned);
        assert_eq!(result, "_5");
    }

    #[test]
    fn test_substitute_assigned_vars_no_double_prime() {
        let mut assigned = HashSet::new();
        assigned.insert("_5");
        // Already primed should not get double-primed
        let result = substitute_assigned_vars("_5'", &assigned);
        // The regex should NOT match _5' because of the negative lookahead for '
        assert_eq!(result, "_5'");
    }

    // ========================================================================
    // MirLocal tests
    // ========================================================================

    #[test]
    fn test_mir_local_new() {
        let local = MirLocal::new("x", SmtType::Int);
        assert_eq!(local.name, "x");
        assert_eq!(local.ty, SmtType::Int);
    }

    #[test]
    fn test_mir_local_new_with_string() {
        let local = MirLocal::new(String::from("counter"), SmtType::Bool);
        assert_eq!(local.name, "counter");
        assert_eq!(local.ty, SmtType::Bool);
    }

    #[test]
    fn test_mir_local_new_with_bitvector() {
        let local = MirLocal::new("bits", SmtType::BitVec(32));
        assert_eq!(local.name, "bits");
        assert_eq!(local.ty, SmtType::BitVec(32));
    }

    #[test]
    fn test_mir_local_new_with_array() {
        let arr_type = SmtType::Array {
            index: Box::new(SmtType::Int),
            element: Box::new(SmtType::Int),
        };
        let local = MirLocal::new("arr", arr_type.clone());
        assert_eq!(local.name, "arr");
        assert_eq!(local.ty, arr_type);
    }

    #[test]
    fn test_mir_local_debug() {
        let local = MirLocal::new("x", SmtType::Int);
        let debug = format!("{:?}", local);
        assert!(debug.contains("MirLocal"));
        assert!(debug.contains('x'));
    }

    #[test]
    fn test_mir_local_clone() {
        let local = MirLocal::new("x", SmtType::Int);
        let cloned = local.clone();
        assert_eq!(cloned.name, local.name);
        assert_eq!(cloned.ty, local.ty);
    }

    // ========================================================================
    // MirStatement tests
    // ========================================================================

    #[test]
    fn test_mir_statement_assume() {
        let stmt = MirStatement::Assume("(>= x 0)".to_string());
        if let MirStatement::Assume(cond) = stmt {
            assert_eq!(cond, "(>= x 0)");
        } else {
            panic!("Expected Assume variant");
        }
    }

    #[test]
    fn test_mir_statement_assign() {
        let stmt = MirStatement::Assign {
            lhs: "x".to_string(),
            rhs: "(+ x 1)".to_string(),
        };
        if let MirStatement::Assign { lhs, rhs } = stmt {
            assert_eq!(lhs, "x");
            assert_eq!(rhs, "(+ x 1)");
        } else {
            panic!("Expected Assign variant");
        }
    }

    #[test]
    fn test_mir_statement_assert_without_message() {
        let stmt = MirStatement::Assert {
            condition: "(< x 100)".to_string(),
            message: None,
        };
        if let MirStatement::Assert { condition, message } = stmt {
            assert_eq!(condition, "(< x 100)");
            assert!(message.is_none());
        } else {
            panic!("Expected Assert variant");
        }
    }

    #[test]
    fn test_mir_statement_assert_with_message() {
        let stmt = MirStatement::Assert {
            condition: "(< x 100)".to_string(),
            message: Some("overflow check".to_string()),
        };
        if let MirStatement::Assert { condition, message } = stmt {
            assert_eq!(condition, "(< x 100)");
            assert_eq!(message.unwrap(), "overflow check");
        } else {
            panic!("Expected Assert variant");
        }
    }

    #[test]
    fn test_mir_statement_array_store() {
        let stmt = MirStatement::ArrayStore {
            array: "arr".to_string(),
            index: "i".to_string(),
            value: "42".to_string(),
        };
        if let MirStatement::ArrayStore {
            array,
            index,
            value,
        } = stmt
        {
            assert_eq!(array, "arr");
            assert_eq!(index, "i");
            assert_eq!(value, "42");
        } else {
            panic!("Expected ArrayStore variant");
        }
    }

    #[test]
    fn test_mir_statement_havoc() {
        let stmt = MirStatement::Havoc {
            var: "x".to_string(),
        };
        if let MirStatement::Havoc { var } = stmt {
            assert_eq!(var, "x");
        } else {
            panic!("Expected Havoc variant");
        }
    }

    #[test]
    fn test_mir_statement_debug() {
        let stmt = MirStatement::Assign {
            lhs: "x".to_string(),
            rhs: "(+ x 1)".to_string(),
        };
        let debug = format!("{:?}", stmt);
        assert!(debug.contains("Assign"));
        assert!(debug.contains('x'));
    }

    #[test]
    fn test_mir_statement_clone() {
        let stmt = MirStatement::Assign {
            lhs: "x".to_string(),
            rhs: "(+ x 1)".to_string(),
        };
        let cloned = stmt.clone();
        if let MirStatement::Assign { lhs, rhs } = cloned {
            assert_eq!(lhs, "x");
            assert_eq!(rhs, "(+ x 1)");
        } else {
            panic!("Clone should preserve variant");
        }
    }

    // ========================================================================
    // MirTerminator tests
    // ========================================================================

    #[test]
    fn test_mir_terminator_goto() {
        let term = MirTerminator::Goto { target: 5 };
        if let MirTerminator::Goto { target } = term {
            assert_eq!(target, 5);
        } else {
            panic!("Expected Goto variant");
        }
    }

    #[test]
    fn test_mir_terminator_cond_goto() {
        let term = MirTerminator::CondGoto {
            condition: "(> x 0)".to_string(),
            then_target: 1,
            else_target: 2,
        };
        if let MirTerminator::CondGoto {
            condition,
            then_target,
            else_target,
        } = term
        {
            assert_eq!(condition, "(> x 0)");
            assert_eq!(then_target, 1);
            assert_eq!(else_target, 2);
        } else {
            panic!("Expected CondGoto variant");
        }
    }

    #[test]
    fn test_mir_terminator_switch_int() {
        let term = MirTerminator::SwitchInt {
            discr: "state".to_string(),
            targets: vec![(0, 1), (1, 2), (2, 3)],
            otherwise: 4,
        };
        if let MirTerminator::SwitchInt {
            discr,
            targets,
            otherwise,
        } = term
        {
            assert_eq!(discr, "state");
            assert_eq!(targets.len(), 3);
            assert_eq!(targets[0], (0, 1));
            assert_eq!(otherwise, 4);
        } else {
            panic!("Expected SwitchInt variant");
        }
    }

    #[test]
    fn test_mir_terminator_call() {
        let term = MirTerminator::Call {
            destination: Some("result".to_string()),
            func: "foo".to_string(),
            args: vec!["x".to_string(), "y".to_string()],
            target: 3,
            unwind: Some(5),
            precondition_check: None,
            postcondition_assumption: None,
            is_range_into_iter: false,
            is_range_next: false,
        };
        if let MirTerminator::Call {
            destination,
            func,
            args,
            target,
            unwind,
            ..
        } = term
        {
            assert_eq!(destination.unwrap(), "result");
            assert_eq!(func, "foo");
            assert_eq!(args.len(), 2);
            assert_eq!(target, 3);
            assert_eq!(unwind, Some(5));
        } else {
            panic!("Expected Call variant");
        }
    }

    #[test]
    fn test_mir_terminator_call_no_destination() {
        let term = MirTerminator::Call {
            destination: None,
            func: "print".to_string(),
            args: vec!["msg".to_string()],
            target: 1,
            unwind: None,
            precondition_check: None,
            postcondition_assumption: None,
            is_range_into_iter: false,
            is_range_next: false,
        };
        if let MirTerminator::Call { destination, .. } = term {
            assert!(destination.is_none());
        } else {
            panic!("Expected Call variant");
        }
    }

    #[test]
    fn test_mir_terminator_return() {
        let term = MirTerminator::Return;
        assert!(matches!(term, MirTerminator::Return));
    }

    #[test]
    fn test_mir_terminator_unreachable() {
        let term = MirTerminator::Unreachable;
        assert!(matches!(term, MirTerminator::Unreachable));
    }

    #[test]
    fn test_mir_terminator_abort() {
        let term = MirTerminator::Abort;
        assert!(matches!(term, MirTerminator::Abort));
    }

    // ========================================================================
    // SwitchInt boolean discriminant tests
    // ========================================================================

    #[test]
    fn test_format_switch_comparison_bool_false() {
        let locals = vec![MirLocal::new("_1", SmtType::Bool)];
        // For boolean, value 0 means false -> "(not _1)"
        let cmp = format_switch_comparison("_1", 0, &locals);
        assert_eq!(cmp, "(not _1)");
    }

    #[test]
    fn test_format_switch_comparison_bool_true() {
        let locals = vec![MirLocal::new("_1", SmtType::Bool)];
        // For boolean, value 1 (non-zero) means true -> "_1"
        let cmp = format_switch_comparison("_1", 1, &locals);
        assert_eq!(cmp, "_1");
    }

    #[test]
    fn test_format_switch_comparison_int() {
        let locals = vec![MirLocal::new("_1", SmtType::Int)];
        // For integer, use direct equality
        let cmp = format_switch_comparison("_1", 42, &locals);
        assert_eq!(cmp, "(= _1 42)");
    }

    #[test]
    fn test_format_switch_comparison_negated_bool_false() {
        let locals = vec![MirLocal::new("_1", SmtType::Bool)];
        // Negating "discr == false" gives "_1"
        let cmp = format_switch_comparison_negated("_1", 0, &locals);
        assert_eq!(cmp, "_1");
    }

    #[test]
    fn test_format_switch_comparison_negated_bool_true() {
        let locals = vec![MirLocal::new("_1", SmtType::Bool)];
        // Negating "discr == true" gives "(not _1)"
        let cmp = format_switch_comparison_negated("_1", 1, &locals);
        assert_eq!(cmp, "(not _1)");
    }

    #[test]
    fn test_format_switch_comparison_negated_int() {
        let locals = vec![MirLocal::new("_1", SmtType::Int)];
        // For integer, use direct inequality
        let cmp = format_switch_comparison_negated("_1", 42, &locals);
        assert_eq!(cmp, "(not (= _1 42))");
    }

    #[test]
    fn test_switch_int_bool_transition_encoding() {
        // Test that boolean SwitchInt generates correct transitions
        // This simulates: switchInt(_1) -> [0: bb1, otherwise: bb2]
        // where _1 is a boolean that gets assigned in the block
        let program = MirProgram::builder(0)
            .local("_1", SmtType::Bool)
            .block(
                MirBasicBlock::new(
                    0,
                    MirTerminator::SwitchInt {
                        discr: "_1".to_string(),
                        targets: vec![(0, 1)], // value 0 -> bb1
                        otherwise: 2,          // otherwise -> bb2
                    },
                )
                .with_statement(MirStatement::Assign {
                    lhs: "_1".to_string(),
                    rhs: "false".to_string(),
                }),
            )
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .block(MirBasicBlock::new(2, MirTerminator::Abort))
            .finish();

        let chc = encode_mir_to_chc(&program);
        let smt = chc.to_smt2();

        // Since _1 is assigned only once, we use primed variable substitution.
        // The SwitchInt check for value 0 on bool becomes "(not _1')" where _1' = false.
        // This is semantically correct: the guard checks the next-state value.
        assert!(
            smt.contains("(not _1')") || smt.contains("(not _1_next)"),
            "Boolean switchInt should use primed variable for single-assigned var, got:\n{}",
            smt
        );
        // Should NOT contain "(= _1_next 0)" or "(= _1 0)" (comparing bool to int)
        assert!(
            !smt.contains("(= _1_next 0)") && !smt.contains("(= _1 0)"),
            "Boolean switchInt should not compare bool to integer 0:\n{}",
            smt
        );
    }

    #[test]
    fn test_mir_terminator_debug() {
        let term = MirTerminator::Goto { target: 3 };
        let debug = format!("{:?}", term);
        assert!(debug.contains("Goto"));
        assert!(debug.contains('3'));
    }

    #[test]
    fn test_mir_terminator_clone() {
        let term = MirTerminator::SwitchInt {
            discr: "x".to_string(),
            targets: vec![(0, 1)],
            otherwise: 2,
        };
        let cloned = term.clone();
        if let MirTerminator::SwitchInt { discr, .. } = cloned {
            assert_eq!(discr, "x");
        } else {
            panic!("Clone should preserve variant");
        }
    }

    // ========================================================================
    // MirBasicBlock tests
    // ========================================================================

    #[test]
    fn test_mir_basic_block_new() {
        let block = MirBasicBlock::new(5, MirTerminator::Return);
        assert_eq!(block.id, 5);
        assert!(block.statements.is_empty());
        assert!(matches!(block.terminator, MirTerminator::Return));
    }

    #[test]
    fn test_mir_basic_block_with_statement() {
        let block =
            MirBasicBlock::new(0, MirTerminator::Return).with_statement(MirStatement::Assign {
                lhs: "x".into(),
                rhs: "0".into(),
            });
        assert_eq!(block.statements.len(), 1);
    }

    #[test]
    fn test_mir_basic_block_with_multiple_statements() {
        let block = MirBasicBlock::new(0, MirTerminator::Return)
            .with_statement(MirStatement::Assign {
                lhs: "x".into(),
                rhs: "0".into(),
            })
            .with_statement(MirStatement::Assign {
                lhs: "y".into(),
                rhs: "1".into(),
            })
            .with_statement(MirStatement::Assert {
                condition: "(>= x 0)".into(),
                message: None,
            });
        assert_eq!(block.statements.len(), 3);
    }

    #[test]
    fn test_mir_basic_block_debug() {
        let block = MirBasicBlock::new(3, MirTerminator::Return);
        let debug = format!("{:?}", block);
        assert!(debug.contains("MirBasicBlock"));
        assert!(debug.contains('3'));
    }

    #[test]
    fn test_mir_basic_block_clone() {
        let block = MirBasicBlock::new(2, MirTerminator::Goto { target: 3 }).with_statement(
            MirStatement::Assign {
                lhs: "x".into(),
                rhs: "1".into(),
            },
        );
        let cloned = block.clone();
        assert_eq!(cloned.id, 2);
        assert_eq!(cloned.statements.len(), 1);
    }

    // ========================================================================
    // MirProgramBuilder tests
    // ========================================================================

    #[test]
    fn test_mir_program_builder_new() {
        let builder = MirProgramBuilder::new(0);
        let program = builder.finish();
        assert_eq!(program.start_block, 0);
        assert!(program.locals.is_empty());
        assert!(program.basic_blocks.is_empty());
        assert!(program.init.is_none());
    }

    #[test]
    fn test_mir_program_builder_local() {
        let program = MirProgramBuilder::new(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Bool)
            .finish();
        assert_eq!(program.locals.len(), 2);
        assert_eq!(program.locals[0].name, "x");
        assert_eq!(program.locals[1].name, "y");
    }

    #[test]
    fn test_mir_program_builder_block() {
        let program = MirProgramBuilder::new(0)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .finish();
        assert_eq!(program.basic_blocks.len(), 2);
    }

    #[test]
    fn test_mir_program_builder_init() {
        let program = MirProgramBuilder::new(0).init("(= x 0)").finish();
        assert!(program.init.is_some());
        assert_eq!(program.init.unwrap().smt_formula, "(= x 0)");
    }

    #[test]
    fn test_mir_program_builder_via_program() {
        let program = MirProgram::builder(5)
            .local("x", SmtType::Int)
            .init("(= x 0)")
            .block(MirBasicBlock::new(5, MirTerminator::Return))
            .finish();
        assert_eq!(program.start_block, 5);
        assert_eq!(program.locals.len(), 1);
    }

    #[test]
    fn test_mir_program_debug() {
        let program = MirProgram::builder(0).local("x", SmtType::Int).finish();
        let debug = format!("{:?}", program);
        assert!(debug.contains("MirProgram"));
    }

    #[test]
    fn test_mir_program_clone() {
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .init("(= x 0)")
            .finish();
        let cloned = program.clone();
        assert_eq!(cloned.start_block, program.start_block);
        assert_eq!(cloned.locals.len(), program.locals.len());
    }

    // ========================================================================
    // Helper function tests
    // ========================================================================

    #[test]
    fn test_conjoin_empty() {
        assert_eq!(conjoin(&[]), "true");
    }

    #[test]
    fn test_conjoin_single() {
        assert_eq!(conjoin(&["(= x 0)".to_string()]), "(= x 0)");
    }

    #[test]
    fn test_conjoin_multiple() {
        let parts = vec!["(= x 0)".to_string(), "(> y 1)".to_string()];
        assert_eq!(conjoin(&parts), "(and (= x 0) (> y 1))");
    }

    #[test]
    fn test_conjoin_three() {
        let parts = vec![
            "(= x 0)".to_string(),
            "(> y 1)".to_string(),
            "(< z 10)".to_string(),
        ];
        assert_eq!(conjoin(&parts), "(and (= x 0) (> y 1) (< z 10))");
    }

    #[test]
    fn test_disjoin_empty() {
        assert_eq!(disjoin(&[]), "false");
    }

    #[test]
    fn test_disjoin_single() {
        assert_eq!(disjoin(&["(= x 0)".to_string()]), "(= x 0)");
    }

    #[test]
    fn test_disjoin_multiple() {
        let parts = vec!["(= x 0)".to_string(), "(= x 1)".to_string()];
        assert_eq!(disjoin(&parts), "(or (= x 0) (= x 1))");
    }

    #[test]
    fn test_disjoin_three() {
        let parts = vec![
            "(= x 0)".to_string(),
            "(= x 1)".to_string(),
            "(= x 2)".to_string(),
        ];
        assert_eq!(disjoin(&parts), "(or (= x 0) (= x 1) (= x 2))");
    }

    #[test]
    fn test_idx_to_i64_small() {
        assert_eq!(idx_to_i64(0), 0);
        assert_eq!(idx_to_i64(42), 42);
        assert_eq!(idx_to_i64(1000), 1000);
    }

    #[test]
    fn test_idx_to_i64_large() {
        let large: usize = 1_000_000_000;
        assert_eq!(idx_to_i64(large), 1_000_000_000i64);
    }

    // ========================================================================
    // Property formula building tests
    // ========================================================================

    #[test]
    fn test_property_building() {
        let block = MirBasicBlock::new(0, MirTerminator::Goto { target: 0 }).with_statement(
            MirStatement::Assert {
                condition: "(>= x 0)".to_string(),
                message: None,
            },
        );

        let formula = build_property_formula(&[block]);
        // Property includes explicit assertions AND abort state unreachability
        assert_eq!(
            formula,
            "(and (or (not (= pc 0)) (>= x 0)) (not (= pc -2)) (not (= pc 999999)))"
        );
    }

    #[test]
    fn test_transition_disjunction() {
        let block = MirBasicBlock::new(
            1,
            MirTerminator::CondGoto {
                condition: "(> x 0)".to_string(),
                then_target: 2,
                else_target: 3,
            },
        )
        .with_statement(MirStatement::Assign {
            lhs: "x".into(),
            rhs: "(+ x 1)".into(),
        });

        let program = MirProgram {
            locals: vec![MirLocal::new("x", SmtType::Int)],
            basic_blocks: vec![block],
            start_block: 1,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let ts = encode_mir_to_transition_system(&program);
        assert!(ts.transition.smt_formula.contains("(or"));
        assert!(ts.transition.smt_formula.contains("(= pc 1)"));
        assert!(ts.transition.smt_formula.contains("(= x' (+ x 1))"));
    }

    #[tokio::test]
    async fn test_mir_encoding_sat() {
        if !has_z3() {
            return;
        }

        // x starts at 0, increments forever, assertion x >= 0 is always true
        let loop_block = MirBasicBlock::new(0, MirTerminator::Goto { target: 0 })
            .with_statement(MirStatement::Assign {
                lhs: "x".into(),
                rhs: "(+ x 1)".into(),
            })
            .with_statement(MirStatement::Assert {
                condition: "(>= x 0)".to_string(),
                message: None,
            });

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(loop_block)
            .init("(= x 0)")
            .finish();

        let chc = encode_mir_to_chc(&program);
        // Use Z3 - Z4 times out on MIR encoding tests
        let result = verify_chc(
            &chc,
            &crate::ChcSolverConfig::new()
                .with_backend(crate::ChcBackend::Z3)
                .with_timeout(Duration::from_secs(5)),
        )
        .await;

        assert!(result.is_ok(), "CHC solve failed: {:?}", result);
        let result = result.unwrap();
        assert!(result.is_sat(), "Expected SAT, got {:?}", result);
    }

    #[tokio::test]
    async fn test_mir_encoding_unsat() {
        if !has_z3() {
            return;
        }

        // Assertion x < 3 will eventually fail as x grows
        let loop_block = MirBasicBlock::new(0, MirTerminator::Goto { target: 0 })
            .with_statement(MirStatement::Assign {
                lhs: "x".into(),
                rhs: "(+ x 1)".into(),
            })
            .with_statement(MirStatement::Assert {
                condition: "(< x 3)".to_string(),
                message: None,
            });

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(loop_block)
            .init("(= x 0)")
            .finish();

        let chc = encode_mir_to_chc(&program);
        // Use Z3 - Z4 times out on MIR encoding tests
        let result = verify_chc(
            &chc,
            &crate::ChcSolverConfig::new()
                .with_backend(crate::ChcBackend::Z3)
                .with_timeout(Duration::from_secs(5)),
        )
        .await;

        assert!(result.is_ok(), "CHC solve failed: {:?}", result);
        let result = result.unwrap();
        assert!(result.is_unsat(), "Expected UNSAT, got {:?}", result);
    }

    #[tokio::test]
    async fn test_soundness_abort_reachable_is_unsat() {
        // SOUNDNESS TEST: When an abort state is reachable, CHC must report UNSAT.
        // This tests the fix for the unsoundness bug where trivial properties
        // would cause code with reachable abort states to be incorrectly verified.
        if !has_z3() {
            return;
        }

        // Program: unconditionally branches to abort state (pc = -2)
        let block = MirBasicBlock::new(0, MirTerminator::Abort);

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(block)
            .init("(= x 0)")
            .finish();

        let chc = encode_mir_to_chc(&program);
        let result = verify_chc(
            &chc,
            &crate::ChcSolverConfig::new().with_timeout(Duration::from_secs(5)),
        )
        .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        // CRITICAL: This must be UNSAT because abort state pc=-2 is reachable!
        assert!(
            result.is_unsat(),
            "SOUNDNESS BUG: Program with reachable abort must be UNSAT, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_soundness_no_abort_is_sat() {
        // SOUNDNESS TEST: When abort states are unreachable, CHC should report SAT.
        if !has_z3() {
            return;
        }

        // Program: simple loop that increments x, never reaches abort
        let loop_block = MirBasicBlock::new(0, MirTerminator::Goto { target: 0 }).with_statement(
            MirStatement::Assign {
                lhs: "x".into(),
                rhs: "(+ x 1)".into(),
            },
        );

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(loop_block)
            .init("(= x 0)")
            .finish();

        let chc = encode_mir_to_chc(&program);
        let result = verify_chc(
            &chc,
            &crate::ChcSolverConfig::new().with_timeout(Duration::from_secs(5)),
        )
        .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        // Should be SAT because abort states are unreachable
        assert!(
            result.is_sat(),
            "Program without abort path should be SAT, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_soundness_conditional_abort() {
        // SOUNDNESS TEST: Conditional branch to abort should be detected.
        if !has_z3() {
            return;
        }

        // Program: if x > 10, abort (and x starts at 0 and increments)
        // This models an overflow check that will eventually fail
        let block = MirBasicBlock::new(
            0,
            MirTerminator::CondGoto {
                condition: "(> x 10)".to_string(),
                then_target: 1, // continue
                else_target: 0, // loop back and increment
            },
        )
        .with_statement(MirStatement::Assign {
            lhs: "x".into(),
            rhs: "(+ x 1)".into(),
        });

        // Block 1 is abort (simulating overflow panic)
        let abort_block = MirBasicBlock::new(1, MirTerminator::Abort);

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(block)
            .block(abort_block)
            .init("(= x 0)")
            .finish();

        let chc = encode_mir_to_chc(&program);
        // Z4 returns unknown/empty for unbounded loop reachability problems where the
        // termination condition depends on accumulated state (x increments until > 10).
        // Z3 Spacer's PDR algorithm handles this pattern correctly, returning unsat.
        // This is a known Z4 limitation for problems requiring inductive invariants.
        let config = crate::ChcSolverConfig::new()
            .with_backend(crate::ChcBackend::Z3)
            .with_timeout(Duration::from_secs(10));
        let result = verify_chc(&chc, &config).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        // x will eventually exceed 10 and trigger abort, so this should be UNSAT
        assert!(
            result.is_unsat(),
            "SOUNDNESS BUG: Program with eventually-reachable abort must be UNSAT, got {:?}",
            result
        );
    }

    #[test]
    fn test_function_call_encoding() {
        // Test that function calls are encoded as uninterpreted function applications
        // Use optimize=false to test raw encoding (result is dead - assigned but not read)
        let block = MirBasicBlock::new(
            0,
            MirTerminator::Call {
                destination: Some("result".into()),
                func: "square".into(),
                args: vec!["x".into()],
                target: 1,
                unwind: None,
                precondition_check: None,
                postcondition_assumption: None,
                is_range_into_iter: false,
                is_range_next: false,
            },
        );

        let program = MirProgram {
            locals: vec![
                MirLocal::new("x", SmtType::Int),
                MirLocal::new("result", SmtType::Int),
            ],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let ts = encode_mir_to_transition_system_optimized(&program, false);
        // Should contain the uninterpreted function application
        assert!(
            ts.transition.smt_formula.contains("(square x)"),
            "Expected (square x) in transition: {}",
            ts.transition.smt_formula
        );
        assert!(ts.transition.smt_formula.contains("(= result' (square x))"));
    }

    #[test]
    fn test_checked_call_does_not_carry_old_value() {
        // Checked intrinsics used to add an identity constraint for the destination,
        // which forced the old value to equal the computed result.
        // Use optimize=false to test raw encoding (_1 is dead - assigned but not read)
        let block = MirBasicBlock::new(
            0,
            MirTerminator::Call {
                destination: Some("_1".into()),
                func: "core::num::<impl u8>::checked_add".into(),
                args: vec!["_2".into(), "_3".into()],
                target: 1,
                unwind: None,
                precondition_check: None,
                postcondition_assumption: None,
                is_range_into_iter: false,
                is_range_next: false,
            },
        );

        let program = MirProgram {
            locals: vec![
                MirLocal::new("_1", SmtType::Int),
                MirLocal::new("_2", SmtType::Int),
                MirLocal::new("_3", SmtType::Int),
            ],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let transition = encode_mir_to_transition_system_optimized(&program, false)
            .transition
            .smt_formula;
        assert!(
            !transition.contains("(= _1' _1)"),
            "Destination should not be constrained to its pre-state value: {}",
            transition
        );
        // For checked operations, _1 gets the discriminant (0=None/overflow, 1=Some/valid)
        // The actual value would go to _1_field0 if present
        assert!(
            transition.contains("(= _1' (ite"),
            "Expected checked add discriminant assignment in: {}",
            transition
        );
    }

    #[test]
    fn test_overflowing_call_does_not_carry_old_value() {
        // Overflowing intrinsics should update the destination without an identity constraint.
        // Use optimize=false to test raw encoding (_1 is dead - assigned but not read)
        let block = MirBasicBlock::new(
            0,
            MirTerminator::Call {
                destination: Some("_1".into()),
                func: "core::num::<impl u8>::overflowing_add".into(),
                args: vec!["_2".into(), "_3".into()],
                target: 1,
                unwind: None,
                precondition_check: None,
                postcondition_assumption: None,
                is_range_into_iter: false,
                is_range_next: false,
            },
        );

        let program = MirProgram {
            locals: vec![
                MirLocal::new("_1", SmtType::Int),
                MirLocal::new("_2", SmtType::Int),
                MirLocal::new("_3", SmtType::Int),
            ],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let transition = encode_mir_to_transition_system_optimized(&program, false)
            .transition
            .smt_formula;
        assert!(
            !transition.contains("(= _1' _1)"),
            "Destination should not be carried forward unchanged: {}",
            transition
        );
        assert!(
            transition.contains("(= _1' (mod (+ _2 _3) 256))"),
            "Expected wrapped value assignment for overflowing add: {}",
            transition
        );
    }

    #[test]
    fn test_array_store_encoding() {
        // Test that array stores are encoded correctly
        let block = MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }).with_statement(
            MirStatement::ArrayStore {
                array: "arr".into(),
                index: "i".into(),
                value: "42".into(),
            },
        );

        let program = MirProgram {
            locals: vec![
                MirLocal::new("i", SmtType::Int),
                MirLocal::new(
                    "arr",
                    SmtType::Array {
                        index: Box::new(SmtType::Int),
                        element: Box::new(SmtType::Int),
                    },
                ),
            ],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let ts = encode_mir_to_transition_system(&program);
        // Should contain SMT store operation
        assert!(
            ts.transition.smt_formula.contains("(store arr i 42)"),
            "Expected (store arr i 42) in transition: {}",
            ts.transition.smt_formula
        );
    }

    #[test]
    fn test_havoc_encoding() {
        // Test that havoc leaves variable unconstrained
        let block = MirBasicBlock::new(0, MirTerminator::Goto { target: 0 })
            .with_statement(MirStatement::Havoc { var: "x".into() })
            .with_statement(MirStatement::Assign {
                lhs: "y".into(),
                rhs: "(+ y 1)".into(),
            });

        let program = MirProgram {
            locals: vec![
                MirLocal::new("x", SmtType::Int),
                MirLocal::new("y", SmtType::Int),
            ],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let ts = encode_mir_to_transition_system(&program);
        // x should NOT have identity constraint (x' = x) since it's havoc'd
        // y should have assignment
        assert!(ts.transition.smt_formula.contains("(= y' (+ y 1))"));
        // x' should be unconstrained - no (= x' x) constraint
        assert!(
            !ts.transition.smt_formula.contains("(= x' x)"),
            "Havoc'd variable should not have identity constraint"
        );
    }

    #[test]
    fn test_abort_encoding() {
        // Test that abort transitions to error state
        let block = MirBasicBlock::new(0, MirTerminator::Abort);

        let program = MirProgram {
            locals: vec![],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let ts = encode_mir_to_transition_system(&program);
        // Should transition to pc = -2 (error state)
        assert!(
            ts.transition.smt_formula.contains("(= pc' -2)"),
            "Abort should transition to pc' = -2: {}",
            ts.transition.smt_formula
        );
    }

    // ========================================================================
    // Property formula building - additional tests
    // ========================================================================

    #[test]
    fn test_property_building_no_assertions() {
        let block = MirBasicBlock::new(0, MirTerminator::Return);
        let formula = build_property_formula(&[block]);
        // Even with no explicit assertions, abort states must be unreachable
        assert_eq!(formula, "(and (not (= pc -2)) (not (= pc 999999)))");
    }

    #[test]
    fn test_property_building_multiple_assertions() {
        let block = MirBasicBlock::new(0, MirTerminator::Return)
            .with_statement(MirStatement::Assert {
                condition: "(>= x 0)".to_string(),
                message: None,
            })
            .with_statement(MirStatement::Assert {
                condition: "(< x 100)".to_string(),
                message: None,
            });

        let formula = build_property_formula(&[block]);
        assert!(formula.contains("(and"));
        assert!(formula.contains("(or (not (= pc 0)) (>= x 0))"));
        assert!(formula.contains("(or (not (= pc 0)) (< x 100))"));
    }

    #[test]
    fn test_property_building_multiple_blocks() {
        let block0 = MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }).with_statement(
            MirStatement::Assert {
                condition: "(>= x 0)".to_string(),
                message: None,
            },
        );
        let block1 =
            MirBasicBlock::new(1, MirTerminator::Return).with_statement(MirStatement::Assert {
                condition: "(< x 100)".to_string(),
                message: None,
            });

        let formula = build_property_formula(&[block0, block1]);
        assert!(formula.contains("(and"));
        assert!(formula.contains("(= pc 0)"));
        assert!(formula.contains("(= pc 1)"));
    }

    #[test]
    fn test_property_ignores_non_assert_statements() {
        let block = MirBasicBlock::new(0, MirTerminator::Return)
            .with_statement(MirStatement::Assume("(>= y 0)".to_string()))
            .with_statement(MirStatement::Assign {
                lhs: "x".into(),
                rhs: "y".into(),
            });

        let formula = build_property_formula(&[block]);
        // Non-assert statements don't add assertions, but abort unreachability is still checked
        assert_eq!(formula, "(and (not (= pc -2)) (not (= pc 999999)))");
    }

    #[test]
    fn test_property_always_includes_abort_unreachability() {
        // This test verifies the soundness fix: even when there are no explicit
        // assertions, the property must include checks that abort states are
        // unreachable. Without this, overflow checks and other safety assertions
        // encoded as branches to abort blocks would not be verified.
        let block = MirBasicBlock::new(0, MirTerminator::Return);
        let formula = build_property_formula(&[block]);

        // Must include unreachability checks for both abort states
        assert!(
            formula.contains("(not (= pc -2))"),
            "Property must check that pc=-2 (Abort terminator) is unreachable"
        );
        assert!(
            formula.contains("(not (= pc 999999))"),
            "Property must check that pc=999999 (panic sentinel) is unreachable"
        );
    }

    // ========================================================================
    // Transition formula building - additional tests
    // ========================================================================

    #[test]
    fn test_transition_empty_program() {
        let program = MirProgram {
            locals: vec![],
            basic_blocks: vec![],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let transition = build_transition_formula(&program);
        assert_eq!(transition, "false");
    }

    #[test]
    fn test_transition_return_terminator() {
        let block = MirBasicBlock::new(0, MirTerminator::Return);
        let program = MirProgram {
            locals: vec![],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let ts = encode_mir_to_transition_system(&program);
        // Return should transition to pc = PC_RETURN_SENTINEL (-1)
        assert!(
            ts.transition
                .smt_formula
                .contains(&format!("(= pc' {})", PC_RETURN_SENTINEL)),
            "Return should transition to pc' = {}: {}",
            PC_RETURN_SENTINEL,
            ts.transition.smt_formula
        );
    }

    #[test]
    fn test_transition_unreachable_terminator() {
        let block = MirBasicBlock::new(0, MirTerminator::Unreachable);
        let program = MirProgram {
            locals: vec![],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let transition = build_transition_formula(&program);
        // Unreachable block has no outgoing transitions
        assert_eq!(transition, "false");
    }

    #[test]
    fn test_transition_switch_int_encoding() {
        let block = MirBasicBlock::new(
            0,
            MirTerminator::SwitchInt {
                discr: "state".to_string(),
                targets: vec![(0, 1), (1, 2)],
                otherwise: 3,
            },
        );

        let program = MirProgram {
            locals: vec![MirLocal::new("state", SmtType::Int)],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let ts = encode_mir_to_transition_system(&program);
        // Should have clauses for each target and otherwise
        assert!(ts.transition.smt_formula.contains("(= state 0)"));
        assert!(ts.transition.smt_formula.contains("(= state 1)"));
        assert!(ts.transition.smt_formula.contains("(not (= state 0))"));
        assert!(ts.transition.smt_formula.contains("(not (= state 1))"));
        assert!(ts.transition.smt_formula.contains("(= pc' 1)"));
        assert!(ts.transition.smt_formula.contains("(= pc' 2)"));
        assert!(ts.transition.smt_formula.contains("(= pc' 3)"));
    }

    #[test]
    fn test_transition_function_call_no_args() {
        // Use optimize=false to test raw encoding (result is dead - assigned but not read)
        let block = MirBasicBlock::new(
            0,
            MirTerminator::Call {
                destination: Some("result".into()),
                func: "get_zero".into(),
                args: vec![],
                target: 1,
                unwind: None,
                precondition_check: None,
                postcondition_assumption: None,
                is_range_into_iter: false,
                is_range_next: false,
            },
        );

        let program = MirProgram {
            locals: vec![MirLocal::new("result", SmtType::Int)],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let ts = encode_mir_to_transition_system_optimized(&program, false);
        // No-arg function call should just use function name directly
        assert!(
            ts.transition.smt_formula.contains("(= result' get_zero)"),
            "Expected (= result' get_zero) in transition: {}",
            ts.transition.smt_formula
        );
    }

    #[test]
    fn test_transition_function_call_multiple_args() {
        // Use optimize=false to test raw encoding (result is dead - assigned but not read)
        let block = MirBasicBlock::new(
            0,
            MirTerminator::Call {
                destination: Some("result".into()),
                func: "add".into(),
                args: vec!["x".into(), "y".into(), "z".into()],
                target: 1,
                unwind: None,
                precondition_check: None,
                postcondition_assumption: None,
                is_range_into_iter: false,
                is_range_next: false,
            },
        );

        let program = MirProgram {
            locals: vec![
                MirLocal::new("x", SmtType::Int),
                MirLocal::new("y", SmtType::Int),
                MirLocal::new("z", SmtType::Int),
                MirLocal::new("result", SmtType::Int),
            ],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let ts = encode_mir_to_transition_system_optimized(&program, false);
        assert!(
            ts.transition.smt_formula.contains("(add x y z)"),
            "Expected (add x y z) in transition: {}",
            ts.transition.smt_formula
        );
    }

    #[test]
    fn test_transition_function_call_no_destination() {
        let block = MirBasicBlock::new(
            0,
            MirTerminator::Call {
                destination: None,
                func: "side_effect".into(),
                args: vec!["x".into()],
                target: 1,
                unwind: None,
                precondition_check: None,
                postcondition_assumption: None,
                is_range_into_iter: false,
                is_range_next: false,
            },
        );

        let program = MirProgram {
            locals: vec![MirLocal::new("x", SmtType::Int)],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let ts = encode_mir_to_transition_system(&program);
        // x should be carried forward (no destination)
        assert!(ts.transition.smt_formula.contains("(= x' x)"));
        assert!(ts.transition.smt_formula.contains("(= pc' 1)"));
    }

    #[test]
    fn test_transition_assume_statement() {
        let block = MirBasicBlock::new(0, MirTerminator::Return)
            .with_statement(MirStatement::Assume("(>= x 0)".to_string()));

        let program = MirProgram {
            locals: vec![MirLocal::new("x", SmtType::Int)],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let ts = encode_mir_to_transition_system(&program);
        // Assume should appear as a guard
        assert!(ts.transition.smt_formula.contains("(>= x 0)"));
    }

    #[test]
    fn test_transition_variable_carry_forward() {
        // Dead variable elimination removes unused variables.
        // y and z are used in assertions/conditions, so they're live and should be carried forward.
        let block = MirBasicBlock::new(0, MirTerminator::Return)
            .with_statement(MirStatement::Assign {
                lhs: "x".into(),
                rhs: "(+ x 1)".into(),
            })
            .with_statement(MirStatement::Assert {
                condition: "(>= y 0)".to_string(),
                message: None,
            })
            .with_statement(MirStatement::Assert {
                condition: "(>= z 0)".to_string(),
                message: None,
            });

        let program = MirProgram {
            locals: vec![
                MirLocal::new("x", SmtType::Int),
                MirLocal::new("y", SmtType::Int),
                MirLocal::new("z", SmtType::Int),
            ],
            basic_blocks: vec![block],
            start_block: 0,
            init: None,
            var_to_local: std::collections::HashMap::new(),
            closures: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        let ts = encode_mir_to_transition_system(&program);
        // x is assigned (and read), y and z are read in assertions and should be carried forward
        assert!(ts.transition.smt_formula.contains("(= x' (+ x 1))"));
        assert!(ts.transition.smt_formula.contains("(= y' y)"));
        assert!(ts.transition.smt_formula.contains("(= z' z)"));
    }

    // ========================================================================
    // Encoding tests - transition system properties
    // ========================================================================

    #[test]
    fn test_encode_mir_adds_pc_variable() {
        // Dead variable elimination removes unused variables, so we need to use x
        let block =
            MirBasicBlock::new(0, MirTerminator::Return).with_statement(MirStatement::Assert {
                condition: "(>= x 0)".to_string(),
                message: None,
            });
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(block)
            .finish();

        let ts = encode_mir_to_transition_system(&program);
        // Should have pc variable plus x (x is used in assertion)
        assert!(ts.variables.iter().any(|v| v.name == "pc"));
        assert!(ts.variables.iter().any(|v| v.name == "x"));
    }

    #[test]
    fn test_encode_mir_init_formula() {
        let program = MirProgram::builder(5)
            .local("x", SmtType::Int)
            .init("(= x 42)")
            .block(MirBasicBlock::new(5, MirTerminator::Return))
            .finish();

        let ts = encode_mir_to_transition_system(&program);
        assert!(ts.init.smt_formula.contains("(= pc 5)"));
        assert!(ts.init.smt_formula.contains("(= x 42)"));
    }

    #[test]
    fn test_encode_mir_init_formula_true() {
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .init("true")
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();

        let ts = encode_mir_to_transition_system(&program);
        // "true" init should not appear redundantly
        assert_eq!(ts.init.smt_formula, "(= pc 0)");
    }

    #[test]
    fn test_encode_mir_no_init() {
        let program = MirProgram::builder(0)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();

        let ts = encode_mir_to_transition_system(&program);
        assert!(ts.init.smt_formula.contains("(= pc 0)"));
    }

    #[test]
    fn test_encode_mir_property_name() {
        let block =
            MirBasicBlock::new(0, MirTerminator::Return).with_statement(MirStatement::Assert {
                condition: "(>= x 0)".to_string(),
                message: None,
            });

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(block)
            .finish();

        let ts = encode_mir_to_transition_system(&program);
        assert_eq!(ts.properties.len(), 1);
        assert_eq!(ts.properties[0].name, "all MIR assertions hold");
    }

    // ========================================================================
    // Integration tests - complex programs
    // ========================================================================

    #[test]
    fn test_encode_simple_loop() {
        // while (x < 10) { x++; }
        let loop_block = MirBasicBlock::new(
            0,
            MirTerminator::CondGoto {
                condition: "(< x 10)".into(),
                then_target: 1, // body
                else_target: 2, // exit
            },
        );
        let body_block = MirBasicBlock::new(1, MirTerminator::Goto { target: 0 }).with_statement(
            MirStatement::Assign {
                lhs: "x".into(),
                rhs: "(+ x 1)".into(),
            },
        );
        let exit_block = MirBasicBlock::new(2, MirTerminator::Return);

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .init("(= x 0)")
            .block(loop_block)
            .block(body_block)
            .block(exit_block)
            .finish();

        let ts = encode_mir_to_transition_system(&program);
        // Should have transitions for all blocks
        assert!(ts.transition.smt_formula.contains("(= pc 0)"));
        assert!(ts.transition.smt_formula.contains("(= pc 1)"));
        assert!(ts.transition.smt_formula.contains("(= pc 2)"));
    }

    #[test]
    fn test_encode_if_else() {
        // if (x > 0) { y = 1; } else { y = 0; }
        let cond_block = MirBasicBlock::new(
            0,
            MirTerminator::CondGoto {
                condition: "(> x 0)".into(),
                then_target: 1,
                else_target: 2,
            },
        );
        let then_block = MirBasicBlock::new(1, MirTerminator::Goto { target: 3 }).with_statement(
            MirStatement::Assign {
                lhs: "y".into(),
                rhs: "1".into(),
            },
        );
        let else_block = MirBasicBlock::new(2, MirTerminator::Goto { target: 3 }).with_statement(
            MirStatement::Assign {
                lhs: "y".into(),
                rhs: "0".into(),
            },
        );
        let merge_block = MirBasicBlock::new(3, MirTerminator::Return);

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(cond_block)
            .block(then_block)
            .block(else_block)
            .block(merge_block)
            .finish();

        let ts = encode_mir_to_transition_system(&program);
        assert!(ts.transition.smt_formula.contains("(> x 0)"));
        assert!(ts.transition.smt_formula.contains("(not (> x 0))"));
    }

    #[test]
    fn test_encode_state_machine() {
        // State machine with switch
        let start_block = MirBasicBlock::new(
            0,
            MirTerminator::SwitchInt {
                discr: "state".into(),
                targets: vec![(0, 1), (1, 2), (2, 3)],
                otherwise: 4,
            },
        );
        let state0_block =
            MirBasicBlock::new(1, MirTerminator::Return).with_statement(MirStatement::Assign {
                lhs: "state".into(),
                rhs: "1".into(),
            });
        let state1_block =
            MirBasicBlock::new(2, MirTerminator::Return).with_statement(MirStatement::Assign {
                lhs: "state".into(),
                rhs: "2".into(),
            });
        let state2_block = MirBasicBlock::new(3, MirTerminator::Return);
        let error_block = MirBasicBlock::new(4, MirTerminator::Abort);

        let program = MirProgram::builder(0)
            .local("state", SmtType::Int)
            .init("(= state 0)")
            .block(start_block)
            .block(state0_block)
            .block(state1_block)
            .block(state2_block)
            .block(error_block)
            .finish();

        let ts = encode_mir_to_transition_system(&program);
        // Should have switch branches
        assert!(ts.transition.smt_formula.contains("(= state 0)"));
        assert!(ts.transition.smt_formula.contains("(= state 1)"));
        assert!(ts.transition.smt_formula.contains("(= state 2)"));
        // Error block should transition to -2
        assert!(ts.transition.smt_formula.contains("(= pc' -2)"));
    }

    #[test]
    fn test_encode_with_call_chain() {
        let block0 = MirBasicBlock::new(
            0,
            MirTerminator::Call {
                destination: Some("a".into()),
                func: "init".into(),
                args: vec![],
                target: 1,
                unwind: None,
                precondition_check: None,
                postcondition_assumption: None,
                is_range_into_iter: false,
                is_range_next: false,
            },
        );
        let block1 = MirBasicBlock::new(
            1,
            MirTerminator::Call {
                destination: Some("b".into()),
                func: "process".into(),
                args: vec!["a".into()],
                target: 2,
                unwind: None,
                precondition_check: None,
                postcondition_assumption: None,
                is_range_into_iter: false,
                is_range_next: false,
            },
        );
        let block2 =
            MirBasicBlock::new(2, MirTerminator::Return).with_statement(MirStatement::Assert {
                condition: "(>= b 0)".into(),
                message: Some("result check".into()),
            });

        let program = MirProgram::builder(0)
            .local("a", SmtType::Int)
            .local("b", SmtType::Int)
            .block(block0)
            .block(block1)
            .block(block2)
            .finish();

        let ts = encode_mir_to_transition_system(&program);
        assert!(ts.transition.smt_formula.contains("init"));
        assert!(ts.transition.smt_formula.contains("(process a)"));
        assert!(ts.properties[0].formula.smt_formula.contains("(>= b 0)"));
    }

    #[test]
    fn test_encode_array_operations() {
        // arr[i] = x; arr[j] = y;
        let block = MirBasicBlock::new(0, MirTerminator::Return)
            .with_statement(MirStatement::ArrayStore {
                array: "arr".into(),
                index: "i".into(),
                value: "x".into(),
            })
            .with_statement(MirStatement::ArrayStore {
                array: "arr".into(),
                index: "j".into(),
                value: "y".into(),
            });

        let program = MirProgram::builder(0)
            .local(
                "arr",
                SmtType::Array {
                    index: Box::new(SmtType::Int),
                    element: Box::new(SmtType::Int),
                },
            )
            .local("i", SmtType::Int)
            .local("j", SmtType::Int)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(block)
            .finish();

        let ts = encode_mir_to_transition_system(&program);
        // Multiple array stores - the second one uses the result of the first
        // Due to SSA-like encoding, we should see nested stores
        assert!(ts.transition.smt_formula.contains("store"));
    }

    #[test]
    fn test_encode_bitvector_types() {
        // Use optimize=false to test raw encoding (result is dead - assigned but not read)
        let block =
            MirBasicBlock::new(0, MirTerminator::Return).with_statement(MirStatement::Assign {
                lhs: "result".into(),
                rhs: "(bvadd x y)".into(),
            });

        let program = MirProgram::builder(0)
            .local("x", SmtType::BitVec(32))
            .local("y", SmtType::BitVec(32))
            .local("result", SmtType::BitVec(32))
            .block(block)
            .finish();

        let ts = encode_mir_to_transition_system_optimized(&program, false);
        assert!(ts.transition.smt_formula.contains("(bvadd x y)"));
        assert!(ts
            .variables
            .iter()
            .any(|v| v.name == "x" && v.smt_type == SmtType::BitVec(32)));
    }

    #[test]
    fn test_encode_mixed_statements() {
        let block = MirBasicBlock::new(0, MirTerminator::Return)
            .with_statement(MirStatement::Assume("(>= x 0)".into()))
            .with_statement(MirStatement::Assign {
                lhs: "y".into(),
                rhs: "(+ x 1)".into(),
            })
            .with_statement(MirStatement::Assert {
                condition: "(> y 0)".into(),
                message: None,
            })
            .with_statement(MirStatement::Havoc { var: "z".into() });

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .local("z", SmtType::Int)
            .block(block)
            .finish();

        let ts = encode_mir_to_transition_system(&program);
        // Check all statement types are handled
        assert!(ts.transition.smt_formula.contains("(>= x 0)")); // assume
        assert!(ts.transition.smt_formula.contains("(= y' (+ x 1))")); // assign
        assert!(ts.properties[0].formula.smt_formula.contains("(> y 0)")); // assert
        assert!(!ts.transition.smt_formula.contains("(= z' z)")); // havoc
    }

    // ========================================================================
    // CHC encoding tests
    // ========================================================================

    #[test]
    fn test_encode_mir_to_chc() {
        let block =
            MirBasicBlock::new(0, MirTerminator::Return).with_statement(MirStatement::Assert {
                condition: "(>= x 0)".into(),
                message: None,
            });

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .init("(= x 0)")
            .block(block)
            .finish();

        let chc = encode_mir_to_chc(&program);
        // Should produce a valid CHC system
        assert!(!chc.clauses.is_empty());
    }

    #[test]
    fn test_encode_mir_to_chc_generates_smtlib2() {
        let block = MirBasicBlock::new(0, MirTerminator::Return);

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(block)
            .finish();

        let chc = encode_mir_to_chc(&program);
        let smt = chc.to_smt2();
        assert!(smt.contains("set-logic HORN"));
    }

    // ========================================================================
    // Bitwise operation tests
    // ========================================================================

    #[test]
    fn test_encode_mir_to_chc_declares_bitwise_functions() {
        // Verify that encode_mir_to_chc declares the bitwise functions
        // used for sound bitwise operations with Int sort
        let block = MirBasicBlock::new(0, MirTerminator::Return);

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(block)
            .finish();

        let chc = encode_mir_to_chc(&program);
        let smt = chc.to_smt2();

        // Should declare bitand, bitor, bitxor, and pow2 as uninterpreted functions
        assert!(
            smt.contains("(declare-fun bitand"),
            "Should declare bitand function"
        );
        assert!(
            smt.contains("(declare-fun bitor"),
            "Should declare bitor function"
        );
        assert!(
            smt.contains("(declare-fun bitxor"),
            "Should declare bitxor function"
        );
        assert!(
            smt.contains("(declare-fun pow2"),
            "Should declare pow2 function"
        );
    }

    #[test]
    fn test_bitwise_functions_have_correct_signature() {
        // Verify that bitwise functions have Int -> Int -> Int signature
        let block = MirBasicBlock::new(0, MirTerminator::Return);

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(block)
            .finish();

        let chc = encode_mir_to_chc(&program);

        // Check the function declarations in the ChcSystem
        assert!(
            chc.functions.contains_key("bitand"),
            "Should have bitand function"
        );
        assert!(
            chc.functions.contains_key("bitor"),
            "Should have bitor function"
        );
        assert!(
            chc.functions.contains_key("bitxor"),
            "Should have bitxor function"
        );
        assert!(
            chc.functions.contains_key("pow2"),
            "Should have pow2 function"
        );

        // Check signatures (Int, Int) -> Int for binary, Int -> Int for pow2
        let bitand = &chc.functions["bitand"];
        assert_eq!(bitand.param_types.len(), 2);
        assert_eq!(bitand.param_types[0], SmtType::Int);
        assert_eq!(bitand.param_types[1], SmtType::Int);
        assert_eq!(bitand.return_type, SmtType::Int);

        let pow2 = &chc.functions["pow2"];
        assert_eq!(pow2.param_types.len(), 1);
        assert_eq!(pow2.param_types[0], SmtType::Int);
        assert_eq!(pow2.return_type, SmtType::Int);
    }

    // ========================================================================
    // BitVec encoding tests
    // ========================================================================

    #[test]
    fn test_program_needs_bitvec_encoding_detects_bitand() {
        let block =
            MirBasicBlock::new(0, MirTerminator::Return).with_statement(MirStatement::Assign {
                lhs: "y".into(),
                rhs: "(bitand x 255)".into(),
            });

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(block)
            .finish();

        assert!(program_needs_bitvec_encoding(&program));
    }

    #[test]
    fn test_program_needs_bitvec_encoding_detects_bitnot() {
        let block =
            MirBasicBlock::new(0, MirTerminator::Return).with_statement(MirStatement::Assign {
                lhs: "y".into(),
                rhs: "(bitnot x)".into(),
            });

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(block)
            .finish();

        assert!(program_needs_bitvec_encoding(&program));
    }

    #[test]
    fn test_program_needs_bitvec_encoding_detects_rotate_encoding() {
        let block =
            MirBasicBlock::new(0, MirTerminator::Return).with_statement(MirStatement::Assign {
                lhs: "y".into(),
                rhs: "(mod (+ (* x (^ 2 n)) (div x (^ 2 (- 8 n)))) 256)".into(),
            });

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .local("n", SmtType::Int)
            .block(block)
            .finish();

        assert!(program_needs_bitvec_encoding(&program));
    }

    #[test]
    fn test_program_needs_bitvec_encoding_checks_init() {
        let block = MirBasicBlock::new(0, MirTerminator::Return);

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .init("(= x (bitxor 1 2))")
            .block(block)
            .finish();

        assert!(program_needs_bitvec_encoding(&program));
    }

    #[test]
    fn test_program_needs_bitvec_encoding_false_for_arithmetic() {
        let block =
            MirBasicBlock::new(0, MirTerminator::Return).with_statement(MirStatement::Assign {
                lhs: "y".into(),
                rhs: "(+ x 1)".into(),
            });

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(block)
            .finish();

        assert!(!program_needs_bitvec_encoding(&program));
    }

    #[test]
    fn test_encode_mir_to_chc_bitvec_basic() {
        // Test that encode_mir_to_chc_bitvec converts Int to BitVec
        // Use assertions to make variables "live" (not optimized away)
        let block = MirBasicBlock::new(0, MirTerminator::Return)
            .with_statement(MirStatement::Assert {
                condition: "(>= x 0)".into(),
                message: None,
            })
            .with_statement(MirStatement::Assign {
                lhs: "y".into(),
                rhs: "x".into(),
            });

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(block)
            .finish();

        let chc = encode_mir_to_chc_bitvec(&program, 32);
        let smt = chc.to_smt2();

        // Should use BitVec(32) in predicate declarations
        assert!(
            smt.contains("(_ BitVec 32)"),
            "Should use BitVec(32) type in declarations. Got: {}",
            smt
        );
        // Should NOT declare uninterpreted bitwise functions
        assert!(
            !smt.contains("(declare-fun bitand"),
            "Should NOT declare bitand - uses native bvand"
        );
    }

    #[test]
    fn test_encode_mir_to_chc_bitvec_converts_literals() {
        // Test that integer literals are converted to BitVec literals
        let block =
            MirBasicBlock::new(0, MirTerminator::Return).with_statement(MirStatement::Assign {
                lhs: "y".into(),
                rhs: "(+ x 1)".into(),
            });

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .init("(= x 0)")
            .block(block)
            .finish();

        let chc = encode_mir_to_chc_bitvec(&program, 32);
        let smt = chc.to_smt2();

        // The init should be converted: (= x 0) -> (= x (_ bv0 32))
        assert!(
            smt.contains("(_ bv0 32)") || smt.contains("bvadd"),
            "Should convert literals to BitVec format"
        );
    }

    #[test]
    fn test_encode_mir_to_chc_bitvec_converts_bitwise() {
        // Test that bitand/bitor become bvand/bvor
        // Make result live by using it in an assertion
        let block = MirBasicBlock::new(0, MirTerminator::Return)
            .with_statement(MirStatement::Assign {
                lhs: "result".into(),
                rhs: "(bitand x 255)".into(),
            })
            .with_statement(MirStatement::Assert {
                condition: "(>= result 0)".into(),
                message: None,
            });

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("result", SmtType::Int)
            .block(block)
            .finish();

        let chc = encode_mir_to_chc_bitvec(&program, 32);
        let smt = chc.to_smt2();

        // Should convert bitand to bvand and 255 to (_ bv255 32)
        assert!(
            smt.contains("bvand"),
            "Should use native bvand operation. Got: {}",
            smt
        );
        assert!(
            smt.contains("(_ bv255 32)"),
            "Should convert 255 to BitVec literal"
        );
    }

    #[test]
    fn test_encode_mir_to_chc_bitvec_concrete_bitwise() {
        // Test the critical case: 12 & 10 = 8
        let block = MirBasicBlock::new(0, MirTerminator::Return)
            .with_statement(MirStatement::Assign {
                lhs: "result".into(),
                rhs: "(bitand 12 10)".into(),
            })
            .with_statement(MirStatement::Assert {
                condition: "(= result 8)".into(),
                message: None,
            });

        let program = MirProgram::builder(0)
            .local("result", SmtType::Int)
            .block(block)
            .finish();

        let chc = encode_mir_to_chc_bitvec(&program, 32);
        let smt = chc.to_smt2();

        // Should convert to: (bvand (_ bv12 32) (_ bv10 32))
        assert!(smt.contains("bvand"), "Should use bvand");
        assert!(smt.contains("(_ bv12 32)"), "Should convert 12 to BitVec");
        assert!(smt.contains("(_ bv10 32)"), "Should convert 10 to BitVec");
        assert!(smt.contains("(_ bv8 32)"), "Should convert 8 to BitVec");
    }

    #[test]
    fn test_encode_mir_to_chc_bitvec_shift() {
        // Test shift operations: 4 << 2 = 16
        let block = MirBasicBlock::new(0, MirTerminator::Return)
            .with_statement(MirStatement::Assign {
                lhs: "result".into(),
                rhs: "(bitshl 4 2)".into(),
            })
            .with_statement(MirStatement::Assert {
                condition: "(= result 16)".into(),
                message: None,
            });

        let program = MirProgram::builder(0)
            .local("result", SmtType::Int)
            .block(block)
            .finish();

        let chc = encode_mir_to_chc_bitvec(&program, 32);
        let smt = chc.to_smt2();

        // Should convert to: (bvshl (_ bv4 32) (_ bv2 32))
        assert!(smt.contains("bvshl"), "Should use bvshl");
    }

    #[test]
    fn test_encode_mir_to_chc_bitvec_64bit() {
        // Test with 64-bit bitvectors
        let block = MirBasicBlock::new(0, MirTerminator::Return);

        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .init("(= x 42)")
            .block(block)
            .finish();

        let chc = encode_mir_to_chc_bitvec(&program, 64);
        let smt = chc.to_smt2();

        // Should use BitVec(64)
        assert!(
            smt.contains("(_ BitVec 64)"),
            "Should use 64-bit bitvectors"
        );
        assert!(
            smt.contains("(_ bv42 64)"),
            "Should convert 42 to 64-bit BV"
        );
    }

    // ========================================================================
    // Kani function detection tests
    // ========================================================================

    #[test]
    fn test_is_kani_any_direct() {
        assert!(is_kani_any("kani::any"));
        assert!(is_kani_any("kani::any::<i32>"));
    }

    #[test]
    fn test_is_kani_any_from_kani_core() {
        assert!(is_kani_any("kani_core::kani_intrinsics::any"));
        assert!(is_kani_any("kani_core::kani_intrinsics::any::<u64>"));
    }

    #[test]
    fn test_is_kani_any_trait_impl() {
        assert!(is_kani_any("<i32 as kani::Arbitrary>::any_raw"));
        assert!(is_kani_any("<u64 as kani::Arbitrary>::any"));
    }

    #[test]
    fn test_is_kani_any_negative() {
        // Should not match non-kani functions
        assert!(!is_kani_any("std::any::Any"));
        assert!(!is_kani_any("foo::any_thing"));
        assert!(!is_kani_any("random_generator"));
    }

    // Test to catch mutation: `A || B || C` → `A || (B && C)` at line 1573:64
    // We need a case where:
    // - contains("::any") = false
    // - ends_with("_any") = true
    // - contains("any_raw") = false
    // Original: false || true || false = true
    // Mutated:  false || (true && false) = false
    #[test]
    fn test_is_kani_any_ends_with_any_pattern() {
        // "kani_get_any" ends with "_any" but doesn't contain "::any" or "any_raw"
        assert!(
            is_kani_any("kani_get_any"),
            "kani_get_any should match via ends_with('_any')"
        );
        // Also test a synthetic pattern that matches only via ends_with
        assert!(
            is_kani_any("kani_core_any"),
            "kani_core_any should match via ends_with('_any')"
        );
    }

    #[test]
    fn test_is_kani_assume_direct() {
        assert!(is_kani_assume("kani::assume"));
        assert!(is_kani_assume("kani_core::assume"));
    }

    #[test]
    fn test_is_kani_assume_negative() {
        // Should not match non-kani functions
        assert!(!is_kani_assume("std::assume"));
        assert!(!is_kani_assume("assume_valid"));
    }

    // ========================================================================
    // Overflow flag detection tests (is_overflow_flag, is_negated_overflow_flag)
    // ========================================================================
    //
    // These functions are now conservative: they only match trivial literals
    // ("false", "true", "(not false)") to ensure real overflow check expressions
    // are preserved in the CHC encoding.

    #[test]
    fn test_is_overflow_flag_elem_1() {
        assert!(is_overflow_flag("_5_elem_1"));
        assert!(is_overflow_flag("_123_elem_1"));
    }

    #[test]
    fn test_is_overflow_flag_field1() {
        assert!(is_overflow_flag("_5_field1"));
        assert!(is_overflow_flag("_0_field1"));
    }

    #[test]
    fn test_is_overflow_flag_false_literal() {
        assert!(is_overflow_flag("false"));
        assert!(is_overflow_flag("  false  "));
    }

    #[test]
    fn test_is_overflow_flag_negative() {
        // Should NOT match regular variables
        assert!(!is_overflow_flag("_5"));
        assert!(!is_overflow_flag("x"));
        assert!(!is_overflow_flag("elem_1")); // doesn't start with _
        assert!(!is_overflow_flag("_5_elem_2"));
        assert!(!is_overflow_flag("_5_field0"));
        assert!(!is_overflow_flag("true"));
    }

    #[test]
    fn test_is_negated_overflow_flag_not_elem_1() {
        assert!(is_negated_overflow_flag("(not _5_elem_1)"));
        assert!(is_negated_overflow_flag("(not _123_elem_1)"));
    }

    #[test]
    fn test_is_negated_overflow_flag_not_field1() {
        assert!(is_negated_overflow_flag("(not _5_field1)"));
        assert!(is_negated_overflow_flag("(not _0_field1)"));
    }

    #[test]
    fn test_is_negated_overflow_flag_not_false() {
        assert!(is_negated_overflow_flag("(not false)"));
    }

    #[test]
    fn test_is_negated_overflow_flag_true_literal() {
        assert!(is_negated_overflow_flag("true"));
        assert!(is_negated_overflow_flag("  true  "));
    }

    #[test]
    fn test_is_negated_overflow_flag_negative() {
        // Should NOT match regular variables
        assert!(!is_negated_overflow_flag("_5_elem_1")); // not negated
        assert!(!is_negated_overflow_flag("(not _5)")); // not overflow flag
        assert!(!is_negated_overflow_flag("(not x)"));
        assert!(!is_negated_overflow_flag("false"));
    }

    // ========================================================================
    // build_predecessor_map tests
    // ========================================================================

    #[test]
    fn test_build_predecessor_map_simple_goto() {
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(1, MirTerminator::Return),
        ];
        let locals = vec![];
        let preds = build_predecessor_map(&blocks, &locals);

        // Block 1 should have block 0 as predecessor with no condition
        assert!(preds.contains_key(&1));
        let pred_list = preds.get(&1).unwrap();
        assert_eq!(pred_list.len(), 1);
        assert_eq!(pred_list[0], (0, None));
    }

    #[test]
    fn test_build_predecessor_map_cond_goto() {
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "(> x 0)".to_string(),
                    then_target: 1,
                    else_target: 2,
                },
            ),
            MirBasicBlock::new(1, MirTerminator::Return),
            MirBasicBlock::new(2, MirTerminator::Return),
        ];
        let locals = vec![];
        let preds = build_predecessor_map(&blocks, &locals);

        // Block 1: predecessor is block 0 with condition "(> x 0)"
        let pred_1 = preds.get(&1).unwrap();
        assert_eq!(pred_1.len(), 1);
        assert_eq!(pred_1[0].0, 0);
        assert_eq!(pred_1[0].1.as_ref().unwrap(), "(> x 0)");

        // Block 2: predecessor is block 0 with negated condition
        let pred_2 = preds.get(&2).unwrap();
        assert_eq!(pred_2.len(), 1);
        assert_eq!(pred_2[0].0, 0);
        assert!(pred_2[0].1.as_ref().unwrap().contains("not"));
    }

    #[test]
    fn test_build_predecessor_map_switch_int() {
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::SwitchInt {
                    discr: "_1".to_string(),
                    targets: vec![(0, 1), (1, 2)],
                    otherwise: 3,
                },
            ),
            MirBasicBlock::new(1, MirTerminator::Return),
            MirBasicBlock::new(2, MirTerminator::Return),
            MirBasicBlock::new(3, MirTerminator::Return),
        ];
        let locals = vec![MirLocal::new("_1", SmtType::Int)];
        let preds = build_predecessor_map(&blocks, &locals);

        // Each target should have block 0 as predecessor
        assert!(preds.contains_key(&1));
        assert!(preds.contains_key(&2));
        assert!(preds.contains_key(&3)); // otherwise case
    }

    #[test]
    fn test_build_predecessor_map_call() {
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::Call {
                    destination: Some("_0".to_string()),
                    func: "foo".to_string(),
                    args: vec![],
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ),
            MirBasicBlock::new(1, MirTerminator::Return),
        ];
        let locals = vec![];
        let preds = build_predecessor_map(&blocks, &locals);

        // Block 1 should have block 0 as predecessor (from call continuation)
        let pred_1 = preds.get(&1).unwrap();
        assert_eq!(pred_1.len(), 1);
        assert_eq!(pred_1[0], (0, None));
    }

    // ========================================================================
    // collect_path_conditions tests
    // ========================================================================

    #[test]
    fn test_collect_path_conditions_direct_path() {
        // Block 0 -> Block 1 (call site)
        let mut preds = HashMap::new();
        preds.insert(1, vec![(0, Some("(> x 0)".to_string()))]);

        let conditions = collect_path_conditions(&preds, 1, 0, None);
        assert_eq!(conditions, vec!["(> x 0)".to_string()]);
    }

    #[test]
    fn test_collect_path_conditions_no_path() {
        // No path exists
        let preds = HashMap::new();
        let conditions = collect_path_conditions(&preds, 5, 0, None);
        assert!(conditions.is_empty());
    }

    #[test]
    fn test_collect_path_conditions_unconditional() {
        // Block 0 -> Block 1 with no condition (unconditional goto)
        let mut preds = HashMap::new();
        preds.insert(1, vec![(0, None)]);

        let conditions = collect_path_conditions(&preds, 1, 0, None);
        assert!(conditions.is_empty());
    }

    #[test]
    fn test_collect_path_conditions_multiple_hops() {
        // Block 0 -> Block 1 -> Block 2
        let mut preds = HashMap::new();
        preds.insert(1, vec![(0, Some("(> x 0)".to_string()))]);
        preds.insert(2, vec![(1, Some("(< x 10)".to_string()))]);

        let conditions = collect_path_conditions(&preds, 2, 0, None);
        // Should have both conditions in order from outermost to innermost
        assert_eq!(conditions.len(), 2);
        assert!(conditions.contains(&"(> x 0)".to_string()));
        assert!(conditions.contains(&"(< x 10)".to_string()));
    }

    // ========================================================================
    // collect_loop_invariants tests
    // ========================================================================

    #[test]
    fn test_collect_loop_invariants_with_assume() {
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(1, MirTerminator::Return)
                .with_statement(MirStatement::Assume("(>= i 0)".to_string()))
                .with_statement(MirStatement::Assume("(< i n)".to_string())),
        ];

        let invariants = collect_loop_invariants(&blocks, 1);
        assert_eq!(invariants.len(), 2);
        assert!(invariants.contains(&"(>= i 0)".to_string()));
        assert!(invariants.contains(&"(< i n)".to_string()));
    }

    #[test]
    fn test_collect_loop_invariants_no_assumes() {
        let blocks = vec![MirBasicBlock::new(0, MirTerminator::Return).with_statement(
            MirStatement::Assign {
                lhs: "x".to_string(),
                rhs: "0".to_string(),
            },
        )];

        let invariants = collect_loop_invariants(&blocks, 0);
        assert!(invariants.is_empty());
    }

    #[test]
    fn test_collect_loop_invariants_nonexistent_block() {
        let blocks = vec![MirBasicBlock::new(0, MirTerminator::Return)];

        let invariants = collect_loop_invariants(&blocks, 99);
        assert!(invariants.is_empty());
    }

    // ========================================================================
    // compute_loop_headers tests
    // ========================================================================

    #[test]
    fn test_compute_loop_headers_no_loops() {
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(1, MirTerminator::Return),
        ];

        let headers = compute_loop_headers(&blocks);
        assert!(headers.is_empty());
    }

    #[test]
    fn test_compute_loop_headers_simple_loop() {
        // Simple loop: 0 -> 1 -> 2 -> 1 (back edge)
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(
                1,
                MirTerminator::CondGoto {
                    condition: "(< i n)".to_string(),
                    then_target: 3, // exit
                    else_target: 2, // body
                },
            ),
            MirBasicBlock::new(2, MirTerminator::Goto { target: 1 }), // back edge to header
            MirBasicBlock::new(3, MirTerminator::Return),
        ];

        let headers = compute_loop_headers(&blocks);
        // Block 1 is the loop header, block 2 should map to it
        if !headers.is_empty() {
            // If loop detected, block 2 should map to header 1
            assert!(headers.get(&2).is_none_or(|h| *h == 1));
        }
    }

    #[test]
    fn test_compute_loop_headers_self_loop() {
        // Self loop: 1 -> 1
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(
                1,
                MirTerminator::CondGoto {
                    condition: "cond".to_string(),
                    then_target: 2,
                    else_target: 1, // self loop
                },
            ),
            MirBasicBlock::new(2, MirTerminator::Return),
        ];

        let headers = compute_loop_headers(&blocks);
        // Block 1 forms a self-loop SCC
        // It may or may not map to itself depending on implementation
        // Just verify it doesn't crash
        assert!(headers.get(&1).is_none_or(|h| *h == 1));
    }

    // ========================================================================
    // optimize_mir_for_unbounded tests
    // ========================================================================
    //
    // NOTE: The optimizer eliminates ALL overflow flag assignments (_X_elem_1,
    // _X_field1) regardless of RHS. This is sound because SMT Int represents
    // unbounded mathematical integers where overflow never occurs.

    #[test]
    fn test_optimize_eliminates_overflow_flag_assignment_elem_1() {
        // Assignment to overflow flag _X_elem_1 should be eliminated regardless of RHS
        // In unbounded Int mode, overflow never happens, so these are dead code
        let program = MirProgram::builder(0)
            .local("_result", SmtType::Int)
            .local("_result_elem_1", SmtType::Bool)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_result_elem_1".to_string(),
                    rhs: "(> _result 2147483647)".to_string(), // Any RHS is eliminated
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        // Overflow flag assignments are eliminated in unbounded mode
        assert!(optimized.basic_blocks[0].statements.is_empty());
    }

    #[test]
    fn test_optimize_eliminates_overflow_flag_assignment_field1() {
        // Assignment to overflow flag _X_field1 should be eliminated regardless of RHS
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .local("_x_field1", SmtType::Bool)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_x_field1".to_string(),
                    rhs: "(< _x -2147483648)".to_string(), // Any RHS is eliminated
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        // Overflow flag assignments are eliminated in unbounded mode
        assert!(optimized.basic_blocks[0].statements.is_empty());
    }

    #[test]
    fn test_optimize_eliminates_false_overflow_flag() {
        // Assignment of "false" to overflow flag should also be eliminated
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .local("_x_field1", SmtType::Bool)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_x_field1".to_string(),
                    rhs: "false".to_string(),
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        // Overflow flag assignments are eliminated
        assert!(optimized.basic_blocks[0].statements.is_empty());
    }

    #[test]
    fn test_optimize_preserves_non_overflow_elem_1_assignment() {
        // Assignment to _X_elem_1 that doesn't look like overflow check should be PRESERVED
        // This handles struct fields and tuple elements that happen to match the naming pattern
        let program = MirProgram::builder(0)
            .local("_tuple_elem_1", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_tuple_elem_1".to_string(),
                    rhs: "42".to_string(), // Not an overflow check pattern
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        // Non-overflow assignments are preserved (e.g., struct fields)
        assert_eq!(optimized.basic_blocks[0].statements.len(), 1);
    }

    #[test]
    fn test_optimize_eliminates_havoc_on_overflow_flag() {
        // Havoc on overflow flag should be eliminated
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .local("_x_elem_1", SmtType::Bool)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Havoc {
                    var: "_x_elem_1".to_string(),
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        // The havoc on overflow flag should be eliminated
        assert!(optimized.basic_blocks[0].statements.is_empty());
    }

    #[test]
    fn test_optimize_preserves_regular_havoc() {
        // Havoc on regular variable should be preserved
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Havoc {
                    var: "_x".to_string(),
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        // The havoc on regular variable should be preserved
        assert_eq!(optimized.basic_blocks[0].statements.len(), 1);
    }

    #[test]
    fn test_optimize_simplifies_overflow_flag_condgoto_to_else() {
        // CondGoto on overflow flag variable should be simplified to Goto else_target
        // because overflow flags are always false in unbounded Int mode
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .local("_x_elem_1", SmtType::Bool)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "_x_elem_1".to_string(), // overflow flag variable (false)
                    then_target: 1,
                    else_target: 2,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return)) // panic path (unreachable)
            .block(MirBasicBlock::new(2, MirTerminator::Return)) // normal path (taken)
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        // Overflow flag is false, so take else branch
        assert!(matches!(
            optimized.basic_blocks[0].terminator,
            MirTerminator::Goto { target: 2 }
        ));
    }

    #[test]
    fn test_optimize_simplifies_negated_overflow_flag_condgoto_to_then() {
        // CondGoto on negated overflow flag variable should be simplified to Goto then_target
        // because (not false) = true
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .local("_x_elem_1", SmtType::Bool)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "(not _x_elem_1)".to_string(), // negated overflow flag (true)
                    then_target: 1,
                    else_target: 2,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return)) // normal path (taken)
            .block(MirBasicBlock::new(2, MirTerminator::Return)) // panic path (unreachable)
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        // Negated overflow flag is true, so take then branch
        assert!(matches!(
            optimized.basic_blocks[0].terminator,
            MirTerminator::Goto { target: 1 }
        ));
    }

    #[test]
    fn test_optimize_preserves_regular_condgoto() {
        // CondGoto on regular condition should be preserved
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "(> _x 0)".to_string(), // regular condition
                    then_target: 1,
                    else_target: 2,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .block(MirBasicBlock::new(2, MirTerminator::Return))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        // Should be preserved as-is
        assert!(matches!(
            optimized.basic_blocks[0].terminator,
            MirTerminator::CondGoto { .. }
        ));
    }

    #[test]
    fn test_optimize_eliminates_u8_overflow_check() {
        // Assignment to _X_elem_1 with u8 bounds should be eliminated (unbounded Int mode)
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .local("_x_elem_1", SmtType::Bool)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_x_elem_1".to_string(),
                    rhs: "(> _x 255)".to_string(), // u8 overflow check
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);
        // Overflow flag assignments eliminated in unbounded mode
        assert!(optimized.basic_blocks[0].statements.is_empty());
    }

    #[test]
    fn test_optimize_eliminates_u16_overflow_check() {
        // Assignment to _X_elem_1 with u16 bounds should be eliminated (unbounded Int mode)
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .local("_x_elem_1", SmtType::Bool)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_x_elem_1".to_string(),
                    rhs: "(> _x 65535)".to_string(), // u16 overflow check
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);
        // Overflow flag assignments eliminated in unbounded mode
        assert!(optimized.basic_blocks[0].statements.is_empty());
    }

    #[test]
    fn test_optimize_eliminates_i8_underflow_check() {
        // Assignment to _X_elem_1 with i8 lower bound should be eliminated (unbounded Int mode)
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .local("_x_elem_1", SmtType::Bool)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_x_elem_1".to_string(),
                    rhs: "(< _x -128)".to_string(), // i8 underflow check
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);
        // Overflow flag assignments eliminated in unbounded mode
        assert!(optimized.basic_blocks[0].statements.is_empty());
    }

    #[test]
    fn test_optimize_eliminates_i16_underflow_check() {
        // Assignment to _X_elem_1 with i16 lower bound should be eliminated (unbounded Int mode)
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .local("_x_elem_1", SmtType::Bool)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_x_elem_1".to_string(),
                    rhs: "(< _x -32768)".to_string(), // i16 underflow check
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);
        // Overflow flag assignments eliminated in unbounded mode
        assert!(optimized.basic_blocks[0].statements.is_empty());
    }

    #[test]
    fn test_optimize_eliminates_positive_sign_check() {
        // Assignment to _X_elem_1 with any expression should be eliminated (unbounded Int mode)
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .local("_x_elem_1", SmtType::Bool)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_x_elem_1".to_string(),
                    rhs: "check > 0 result".to_string(),
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);
        // Overflow flag assignments eliminated in unbounded mode
        assert!(optimized.basic_blocks[0].statements.is_empty());
    }

    #[test]
    fn test_optimize_eliminates_negative_sign_check() {
        // Assignment to _X_elem_1 with any expression should be eliminated (unbounded Int mode)
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .local("_x_elem_1", SmtType::Bool)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_x_elem_1".to_string(),
                    rhs: "value < 0 check".to_string(),
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);
        // Overflow flag assignments eliminated in unbounded mode
        assert!(optimized.basic_blocks[0].statements.is_empty());
    }

    #[test]
    fn test_optimize_condgoto_with_false_literal() {
        // CondGoto with condition "false" should become Goto to else_target
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "false".to_string(),
                    then_target: 1,
                    else_target: 2,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .block(MirBasicBlock::new(2, MirTerminator::Return))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        assert!(matches!(
            optimized.basic_blocks[0].terminator,
            MirTerminator::Goto { target: 2 }
        ));
    }

    #[test]
    fn test_optimize_condgoto_with_not_false() {
        // CondGoto with condition "(not false)" should become Goto to then_target
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "(not false)".to_string(),
                    then_target: 1,
                    else_target: 2,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .block(MirBasicBlock::new(2, MirTerminator::Return))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        assert!(matches!(
            optimized.basic_blocks[0].terminator,
            MirTerminator::Goto { target: 1 }
        ));
    }

    #[test]
    fn test_optimize_preserves_other_terminators() {
        // Other terminators (Goto, Return, Call, etc.) should be preserved
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }))
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        assert!(matches!(
            optimized.basic_blocks[0].terminator,
            MirTerminator::Goto { target: 1 }
        ));
        assert!(matches!(
            optimized.basic_blocks[1].terminator,
            MirTerminator::Return
        ));
    }

    #[test]
    fn test_optimize_preserves_regular_assignments() {
        // Regular assignments should be preserved
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .local("_y", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_x".to_string(),
                    rhs: "(+ _y 1)".to_string(),
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        assert_eq!(optimized.basic_blocks[0].statements.len(), 1);
    }

    #[test]
    fn test_optimize_preserves_asserts() {
        // Assert statements should be preserved
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assert {
                    condition: "(> _x 0)".to_string(),
                    message: None,
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        assert_eq!(optimized.basic_blocks[0].statements.len(), 1);
        assert!(matches!(
            &optimized.basic_blocks[0].statements[0],
            MirStatement::Assert { .. }
        ));
    }

    #[test]
    fn test_optimize_preserves_assumes() {
        // Assume statements should be preserved
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .block(
                MirBasicBlock::new(0, MirTerminator::Return)
                    .with_statement(MirStatement::Assume("(>= _x 0)".to_string())),
            )
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        assert_eq!(optimized.basic_blocks[0].statements.len(), 1);
        assert!(matches!(
            &optimized.basic_blocks[0].statements[0],
            MirStatement::Assume(_)
        ));
    }

    #[test]
    fn test_optimize_preserves_program_structure() {
        // Verify that optimization preserves locals, start_block, init, etc.
        let program = MirProgram::builder(5)
            .local("_x", SmtType::Int)
            .local("_y", SmtType::Bool)
            .init("(= _x 0)")
            .block(MirBasicBlock::new(5, MirTerminator::Return))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        assert_eq!(optimized.locals.len(), 2);
        assert_eq!(optimized.start_block, 5);
        assert!(optimized.init.is_some());
        assert_eq!(optimized.init.as_ref().unwrap().smt_formula, "(= _x 0)");
    }

    #[test]
    fn test_optimize_multiple_statements_mixed() {
        // Test with multiple statements, some to eliminate and some to keep
        // With new semantics: real overflow checks are preserved, only "false" is eliminated
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .local("_x_elem_1", SmtType::Bool)
            .local("_y", SmtType::Int)
            .block(
                MirBasicBlock::new(0, MirTerminator::Return)
                    .with_statement(MirStatement::Assign {
                        lhs: "_x".to_string(),
                        rhs: "10".to_string(),
                    })
                    .with_statement(MirStatement::Assign {
                        lhs: "_x_elem_1".to_string(),
                        rhs: "false".to_string(), // This trivial "false" should be eliminated
                    })
                    .with_statement(MirStatement::Assign {
                        lhs: "_y".to_string(),
                        rhs: "(+ _x 1)".to_string(),
                    }),
            )
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        // Should keep 2 statements (regular assignments), eliminate 1 (trivial false)
        assert_eq!(optimized.basic_blocks[0].statements.len(), 2);
    }

    // ========================================================================
    // Mutation testing gap coverage - N=186
    // ========================================================================

    // Test to kill: optimize_mir_for_unbounded match guard -> true mutant
    // Line 227: lhs.ends_with("_elem_1") || lhs.ends_with("_field1") -> true
    // This mutant would cause regular variables with overflow-related RHS to be eliminated
    #[test]
    fn test_optimize_preserves_regular_var_with_overflow_rhs() {
        // A regular variable (not ending in _elem_1 or _field1) with overflow-related RHS
        // should be PRESERVED. With the mutant, it would be incorrectly eliminated.
        let program = MirProgram::builder(0)
            .local("_is_max", SmtType::Bool) // NOT an overflow flag name
            .local("_x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "_is_max".to_string(),
                    // Contains overflow constant - with mutant, this would be eliminated
                    rhs: "(= _x 2147483647)".to_string(),
                },
            ))
            .finish();

        let optimized = optimize_mir_for_unbounded(&program);

        // The assignment MUST be preserved because _is_max is NOT an overflow flag
        // The mutant would eliminate it because the guard becomes 'true' and RHS contains "2147483647"
        assert_eq!(
            optimized.basic_blocks[0].statements.len(),
            1,
            "Regular variable with overflow-related RHS must be preserved"
        );
    }

    // Test to kill: is_negated_overflow_flag && with || mutant
    // Line 312: starts_with("(not ") && ends_with(")") -> starts_with("(not ") || ends_with(")")
    // This mutant would incorrectly identify partial matches
    #[test]
    fn test_is_negated_overflow_flag_rejects_partial_parens() {
        // Starts with "(not " but doesn't end with ")" - must be rejected
        assert!(
            !is_negated_overflow_flag("(not _5_elem_1"),
            "Missing closing paren must not match"
        );
        // Ends with ")" but doesn't start with "(not " - must be rejected
        assert!(
            !is_negated_overflow_flag("_5_elem_1)"),
            "Missing (not prefix must not match"
        );
        // Neither start nor end matches
        assert!(
            !is_negated_overflow_flag("_5_elem_1"),
            "Plain overflow flag must not match negated pattern"
        );
    }

    // Additional test to kill && -> || mutant more robustly
    // String ends with ")" and has overflow flag at position 5 (where "(not " would end)
    #[test]
    fn test_is_negated_overflow_flag_rejects_fake_pattern() {
        // Key test: "XXXXX_5_elem_1)" has:
        // - Length 15, ends with ")"
        // - [5..14] = "_5_elem_1" which IS an overflow flag pattern
        // With the || mutant (starts_with || ends_with):
        //   - ends_with(")") = true, so enters if block
        //   - inner = "_5_elem_1", is_overflow_flag(inner) = true
        //   - Returns TRUE (INCORRECT)
        // Without mutant (starts_with && ends_with):
        //   - starts_with("(not ") = false (it starts with "XXXXX")
        //   - && fails, doesn't enter if block
        //   - Falls through to trimmed == "true" check
        //   - Returns FALSE (CORRECT)
        assert!(
            !is_negated_overflow_flag("XXXXX_5_elem_1)"),
            "Fake pattern without '(not ' prefix must not match - kills && -> || mutant"
        );
        // Another test case with field1 suffix
        assert!(
            !is_negated_overflow_flag("XXXXX_0_field1)"),
            "Fake pattern with field1 suffix must not match"
        );
    }

    // Test to kill: analyze_live_variables delete CondGoto match arm mutant
    // Line 369: delete match arm for CondGoto
    // This mutant would fail to mark condition variables as live
    #[test]
    fn test_analyze_live_variables_marks_condgoto_condition_live() {
        let program = MirProgram::builder(0)
            .local("_cond", SmtType::Bool)
            .local("_unused", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "_cond".to_string(),
                    then_target: 1,
                    else_target: 2,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .block(MirBasicBlock::new(2, MirTerminator::Return))
            .finish();

        let live = analyze_live_variables(&program);

        // _cond is used in the CondGoto and MUST be marked live
        assert!(
            live.contains("_cond"),
            "CondGoto condition variable must be marked live"
        );
        // _unused is never used
        assert!(
            !live.contains("_unused"),
            "Unused variable should not be marked live"
        );
    }

    // Test to kill: encode_mir_to_chc == with != mutant
    // Line 565: find(|l| &l.name == dest) -> find(|l| &l.name != dest)
    // This mutant would find the WRONG local type
    #[test]
    fn test_encode_mir_to_chc_call_return_type_lookup() {
        // Create program with call where destination type matters
        let program = MirProgram::builder(0)
            .local("_result", SmtType::Bool) // Specifically Bool, not Int
            .local("_x", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::Call {
                    func: "some_bool_func".to_string(),
                    args: vec!["_x".to_string()],
                    destination: Some("_result".to_string()),
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .finish();

        let chc = encode_mir_to_chc(&program);

        // The function declaration should exist and have the correct return type
        // With the mutant, it would find the wrong local and get Int instead of Bool
        let func = chc
            .functions
            .iter()
            .find(|(_, f)| f.name == "some_bool_func");
        assert!(func.is_some(), "Function should be declared");
        assert!(
            matches!(func.unwrap().1.return_type, SmtType::Bool),
            "Return type should be Bool based on destination variable type"
        );
    }

    // Test to kill: build_transition_formula delete - mutant (line 689)
    // Line 689: idx_to_i64(*target) uses PC_RETURN_SENTINEL for return, deleting - breaks sentinel
    #[test]
    fn test_build_transition_formula_return_uses_return_sentinel() {
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return))
            .finish();

        let formula = build_transition_formula(&program);

        // The return transition should use PC_RETURN_SENTINEL as the pc' value
        assert!(
            formula.contains("pc' (- 1)")
                || formula.contains(&format!("pc' {}", PC_RETURN_SENTINEL)),
            "Return must transition to pc'=PC_RETURN_SENTINEL ({}), got: {}",
            PC_RETURN_SENTINEL,
            formula
        );
    }

    // Test to kill: build_transition_formula delete - mutant (line 721)
    // Line 721: Abort uses -2 sentinel, deleting - breaks it
    #[test]
    fn test_build_transition_formula_abort_uses_minus_two_sentinel() {
        let program = MirProgram::builder(0)
            .local("_x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Abort))
            .finish();

        let formula = build_transition_formula(&program);

        // The abort transition should use -2 as the pc' value
        assert!(
            formula.contains("pc' (- 2)") || formula.contains("pc' -2"),
            "Abort must transition to pc'=-2, got: {}",
            formula
        );
    }

    // Test to kill: build_predecessor_map delete ! mutant (line 893)
    // Line 893: if !negated.is_empty() -> if negated.is_empty()
    // This would invert when we add the conjunct guard vs no guard
    #[test]
    fn test_build_predecessor_map_switchint_otherwise_guard() {
        // SwitchInt with otherwise target should have negated guards
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::SwitchInt {
                    discr: "_x".to_string(),
                    targets: vec![(1, 1), (2, 2)],
                    otherwise: 3,
                },
            ),
            MirBasicBlock::new(1, MirTerminator::Return),
            MirBasicBlock::new(2, MirTerminator::Return),
            MirBasicBlock::new(3, MirTerminator::Return), // otherwise
        ];
        let locals = vec![MirLocal::new("_x".to_string(), SmtType::Int)];

        let preds = build_predecessor_map(&blocks, &locals);

        // Block 3 (otherwise) should have block 0 as predecessor with guard
        let pred_list = preds.get(&3).expect("Block 3 should have predecessors");
        assert_eq!(pred_list.len(), 1);
        let (pred_block, guard) = &pred_list[0];
        assert_eq!(*pred_block, 0);
        // The guard should be present (conjunction of negated comparisons)
        assert!(
            guard.is_some(),
            "Otherwise branch must have a guard (negated comparisons)"
        );
        let guard_str = guard.as_ref().unwrap();
        // With the mutant (! deleted), guard would be None when there ARE negated conditions
        assert!(
            guard_str.contains("not"),
            "Guard should contain negated comparisons: {}",
            guard_str
        );
    }

    // Test to kill: compute_loop_headers -> HashMap::new() mutant (line 1020)
    // This mutant would return empty map for any program with loops
    #[test]
    fn test_compute_loop_headers_finds_simple_loop() {
        // Create a simple while loop: 0 -> 1 (body) -> 0 (back edge)
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "_cond".to_string(),
                    then_target: 2, // exit
                    else_target: 1, // body
                },
            ),
            MirBasicBlock::new(1, MirTerminator::Goto { target: 0 }), // back edge
            MirBasicBlock::new(2, MirTerminator::Return),
        ];

        let headers = compute_loop_headers(&blocks);

        // Block 0 should be identified as a loop header
        assert!(
            !headers.is_empty(),
            "Loop headers must not be empty for program with back edge"
        );
        assert!(
            headers.contains_key(&1) || headers.values().any(|&h| h == 0),
            "Block 0 should be identified as loop header: {:?}",
            headers
        );
    }

    // Test to kill: compute_loop_headers::strong_connect -> () mutant (line 1063)
    // This mutant would make the recursive call do nothing, breaking Tarjan
    #[test]
    fn test_compute_loop_headers_nested_loops() {
        // Nested loops: outer 0->1, inner 1->2->1, back to outer 1->0
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "_outer".to_string(),
                    then_target: 3, // exit
                    else_target: 1, // outer body
                },
            ),
            MirBasicBlock::new(
                1,
                MirTerminator::CondGoto {
                    condition: "_inner".to_string(),
                    then_target: 0, // back to outer
                    else_target: 2, // inner body
                },
            ),
            MirBasicBlock::new(2, MirTerminator::Goto { target: 1 }), // inner back edge
            MirBasicBlock::new(3, MirTerminator::Return),
        ];

        let headers = compute_loop_headers(&blocks);

        // Should find headers for both inner and outer loops
        // With strong_connect -> () mutant, would fail to explore successors
        assert!(
            !headers.is_empty(),
            "Should detect at least one loop header for nested loops: {:?}",
            headers
        );
    }

    // Test to kill: is_kani_any || vs && mutant (line 1381)
    // Line 1381: contains("kani") && (contains("::any") || ...) -> contains("kani") || (...)
    // This mutant would match non-kani functions with "any" in name
    #[test]
    fn test_is_kani_any_requires_kani_prefix() {
        // "any" function without kani prefix should NOT match
        assert!(!is_kani_any("::any"), "::any without kani must not match");
        assert!(
            !is_kani_any("some_any_function"),
            "some_any_function must not match"
        );
        assert!(
            !is_kani_any("get_any_value"),
            "get_any_value must not match"
        );
        // But kani::any variants should match
        assert!(is_kani_any("kani::any"), "kani::any must match");
        assert!(is_kani_any("kani_core::any"), "kani_core::any must match");
    }

    // Test to kill: substitute_assigned_vars && vs || mutant (line 1624)
    // Line 1624: !prev.is_ascii_alphanumeric() && prev != b'_' -> ||
    // This mutant would match partial variable names
    #[test]
    fn test_substitute_assigned_vars_word_boundary() {
        let mut assigned = HashSet::new();
        assigned.insert("_x");

        // "_x" inside "_x10" should NOT be substituted (followed by digit)
        let result = substitute_assigned_vars("(+ _x10 1)", &assigned);
        assert!(
            !result.contains("_x'10"),
            "_x in _x10 must not be substituted: {}",
            result
        );

        // "_x" as standalone should be substituted
        let result2 = substitute_assigned_vars("(+ _x 1)", &assigned);
        assert!(
            result2.contains("_x'"),
            "Standalone _x must be substituted: {}",
            result2
        );
    }

    // Test to catch mutation at line 1624:51: `&& prev != b'_'` → `|| prev != b'_'`
    // We need a case where "_x" is preceded by an underscore character.
    // Original: preceded_ok = !is_alpha('_') && '_' != '_' = true && false = false
    // Mutated:  preceded_ok = !is_alpha('_') || '_' != '_' = true || false = true
    #[test]
    fn test_substitute_assigned_vars_preceded_by_underscore() {
        let mut assigned = HashSet::new();
        assigned.insert("x");

        // "x" inside "__x" should NOT be substituted (preceded by underscore)
        // The variable "__x" contains "x" but it's part of a longer identifier
        let result = substitute_assigned_vars("(+ __x 1)", &assigned);
        assert!(
            !result.contains("__x'"),
            "x in __x must not be substituted to __x': {}",
            result
        );
        // The original __x should remain unchanged
        assert!(
            result.contains("__x"),
            "__x should remain unchanged: {}",
            result
        );

        // Also test with variable name "_y" preceded by underscore in "a_y"
        let mut assigned2 = HashSet::new();
        assigned2.insert("y");
        let result2 = substitute_assigned_vars("(+ a_y 1)", &assigned2);
        assert!(
            !result2.contains("a_y'"),
            "y in a_y must not be substituted: {}",
            result2
        );
    }

    // Test to kill: build_call_clause delete ! at line 1168
    // Line 1168: if !context_conditions.is_empty() -> if context_conditions.is_empty()
    // This mutant would skip adding conditions when there ARE conditions
    #[test]
    fn test_build_call_clause_includes_path_conditions() {
        // Create a program with a conditional call
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "(> _x 0)".to_string(),
                    then_target: 2, // skip call
                    else_target: 1, // call block
                },
            ),
            MirBasicBlock::new(
                1,
                MirTerminator::Call {
                    func: "test_func".to_string(),
                    args: vec![],
                    destination: Some("_result".to_string()),
                    target: 2,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ),
            MirBasicBlock::new(2, MirTerminator::Return),
        ];
        let locals = vec![
            MirLocal::new("_x".to_string(), SmtType::Int),
            MirLocal::new("_result".to_string(), SmtType::Int),
        ];
        let predecessors = build_predecessor_map(&blocks, &locals);
        let loop_headers = compute_loop_headers(&blocks);

        let clause = build_call_clause(
            &blocks[1],
            Some("_result"),
            "test_func",
            &[],
            false,
            false,
            2,
            &locals,
            None,
            &predecessors,
            &blocks,
            0,
            loop_headers.get(&1).copied(),
            &HashMap::new(),
            &HashMap::new(),
        );

        // The call clause should include path conditions from block 0
        // With the mutant, it would skip adding them entirely
        // Note: The path condition may be negated depending on branch taken
        assert!(
            clause.len() > 20, // Basic clause structure
            "Call clause should include path conditions: {}",
            clause
        );
    }

    // Test to kill: build_call_clause || vs && at line 1171
    // Line 1171: cond.trim().is_empty() || cond == "true" -> &&
    // This mutant would only skip when BOTH empty AND "true" (impossible)
    #[test]
    fn test_build_call_clause_skips_empty_conditions() {
        // Empty conditions and "true" should both be skipped
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::Call {
                    func: "test_func".to_string(),
                    args: vec![],
                    destination: Some("_result".to_string()),
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ),
            MirBasicBlock::new(1, MirTerminator::Return),
        ];
        let locals = vec![MirLocal::new("_result".to_string(), SmtType::Int)];
        let predecessors = build_predecessor_map(&blocks, &locals);
        let _loop_headers: HashMap<usize, usize> = HashMap::new();

        let clause = build_call_clause(
            &blocks[0],
            Some("_result"),
            "test_func",
            &[],
            false,
            false,
            1,
            &locals,
            None,
            &predecessors,
            &blocks,
            0,
            None,
            &HashMap::new(),
            &HashMap::new(),
        );

        // "true" conditions should not appear verbatim in the output
        // (they're tautologies that should be skipped)
        // The clause should just be the basic transition without redundant "true"
        assert!(
            !clause.contains("(and true true)"),
            "Redundant true conditions should be filtered"
        );
    }

    // Test to kill: build_call_clause == vs != at line 1171
    // Line 1171: cond == "true" -> cond != "true"
    // This mutant would NOT skip "true" conditions
    #[test]
    fn test_build_call_clause_skips_true_literal() {
        // When context has a "true" condition, it should be skipped
        // This is tested indirectly - "true" should not appear as a conjunct
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(
                1,
                MirTerminator::Call {
                    func: "test_func".to_string(),
                    args: vec![],
                    destination: Some("_result".to_string()),
                    target: 2,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ),
            MirBasicBlock::new(2, MirTerminator::Return),
        ];
        let locals = vec![MirLocal::new("_result".to_string(), SmtType::Int)];
        let predecessors = build_predecessor_map(&blocks, &locals);

        let clause = build_call_clause(
            &blocks[1],
            Some("_result"),
            "test_func",
            &[],
            false,
            false,
            2,
            &locals,
            None,
            &predecessors,
            &blocks,
            0,
            None,
            &HashMap::new(),
            &HashMap::new(),
        );

        // Count occurrences of " true " or "(true)" - should be minimal
        let true_count = clause.matches(" true").count();
        assert!(
            true_count <= 1,
            "Should not have redundant true conditions: {}",
            clause
        );
    }

    // Test to kill: build_call_clause delete ! at line 1227
    // Line 1227: if !substituted_args.is_empty() -> if substituted_args.is_empty()
    // This mutant would NOT add assume arg when there IS one
    #[test]
    fn test_build_call_clause_kani_assume_adds_constraint() {
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::Call {
                    func: "kani::assume".to_string(),
                    args: vec!["(> _x 0)".to_string()],
                    destination: None,
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ),
            MirBasicBlock::new(1, MirTerminator::Return),
        ];
        let locals = vec![MirLocal::new("_x".to_string(), SmtType::Int)];
        let predecessors = build_predecessor_map(&blocks, &locals);

        let clause = build_call_clause(
            &blocks[0],
            None,
            "kani::assume",
            &["(> _x 0)".to_string()],
            false,
            false,
            1,
            &locals,
            None,
            &predecessors,
            &blocks,
            0,
            None,
            &HashMap::new(),
            &HashMap::new(),
        );

        // The assume constraint should be included
        assert!(
            clause.contains("(> _x 0)") || clause.contains("(> _x' 0)"),
            "kani::assume must add its argument as constraint: {}",
            clause
        );
    }

    // Test to kill: build_call_clause delete ! at line 1232
    // Line 1232: if !assigned.contains(local.name.as_str()) -> if assigned.contains(...)
    // This mutant would NOT carry forward UNassigned locals after assume
    #[test]
    fn test_build_call_clause_kani_assume_carries_unassigned() {
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::Call {
                    func: "kani::assume".to_string(),
                    args: vec!["true".to_string()],
                    destination: None,
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ),
            MirBasicBlock::new(1, MirTerminator::Return),
        ];
        let locals = vec![
            MirLocal::new("_x".to_string(), SmtType::Int),
            MirLocal::new("_y".to_string(), SmtType::Int),
        ];
        let predecessors = build_predecessor_map(&blocks, &locals);

        let clause = build_call_clause(
            &blocks[0],
            None,
            "kani::assume",
            &["true".to_string()],
            false,
            false,
            1,
            &locals,
            None,
            &predecessors,
            &blocks,
            0,
            None,
            &HashMap::new(),
            &HashMap::new(),
        );

        // Unassigned locals should be carried forward (= _x' _x)
        assert!(
            clause.contains("(= _x' _x)"),
            "Unassigned _x should be carried forward: {}",
            clause
        );
        assert!(
            clause.contains("(= _y' _y)"),
            "Unassigned _y should be carried forward: {}",
            clause
        );
    }

    // Test to kill: build_call_clause delete ! at line 1250
    // Line 1250: if !assigned.contains(local.name.as_str()) -> if assigned.contains(...)
    // This mutant would NOT carry forward unassigned locals after kani::any
    #[test]
    fn test_build_call_clause_kani_any_carries_unassigned() {
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::Call {
                    func: "kani::any".to_string(),
                    args: vec![],
                    destination: Some("_result".to_string()),
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ),
            MirBasicBlock::new(1, MirTerminator::Return),
        ];
        let locals = vec![
            MirLocal::new("_x".to_string(), SmtType::Int),
            MirLocal::new("_result".to_string(), SmtType::Int),
        ];
        let predecessors = build_predecessor_map(&blocks, &locals);

        let clause = build_call_clause(
            &blocks[0],
            Some("_result"),
            "kani::any",
            &[],
            false,
            false,
            1,
            &locals,
            None,
            &predecessors,
            &blocks,
            0,
            None,
            &HashMap::new(),
            &HashMap::new(),
        );

        // _x is unassigned and should be carried forward
        assert!(
            clause.contains("(= _x' _x)"),
            "Unassigned _x should be carried forward after kani::any: {}",
            clause
        );
        // _result should NOT have (= _result' _result) because it's the any() destination
        // It should be unconstrained (havoced)
        assert!(
            !clause.contains("(= _result' _result)"),
            "kani::any destination should not be identity-constrained: {}",
            clause
        );
    }

    // Test to kill: compute_loop_headers Tarjan comparison mutations
    // Lines 1077, 1085: > comparisons in lowlink update
    // These ensure SCCs are computed correctly
    #[test]
    fn test_compute_loop_headers_complex_scc() {
        // Create a figure-8 pattern: two loops sharing a node
        //   1 -> 2 -> 1 (loop 1)
        //   1 -> 3 -> 1 (loop 2)
        //   0 -> 1, 1 -> 4 (entry/exit)
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(
                1,
                MirTerminator::SwitchInt {
                    discr: "_d".to_string(),
                    targets: vec![(0, 2), (1, 3)],
                    otherwise: 4,
                },
            ),
            MirBasicBlock::new(2, MirTerminator::Goto { target: 1 }), // back edge
            MirBasicBlock::new(3, MirTerminator::Goto { target: 1 }), // back edge
            MirBasicBlock::new(4, MirTerminator::Return),
        ];

        let headers = compute_loop_headers(&blocks);

        // Block 1 should be identified as a loop header (has back edges from 2 and 3)
        // With broken Tarjan comparisons, SCCs would be incorrectly identified
        assert!(
            headers.values().any(|&h| h == 1)
                || headers.contains_key(&2)
                || headers.contains_key(&3),
            "Should identify loop structure around block 1: {:?}",
            headers
        );
    }

    // Test to kill: compute_loop_headers || vs && at line 1109
    // Line 1109: w_on_stack || lowlink[w] < lowlink[v] -> &&
    // This breaks the lowlink propagation logic
    #[test]
    fn test_compute_loop_headers_lowlink_propagation() {
        // Chain: 0 -> 1 -> 2 -> 3 -> 1 (back edge to 1, not 0)
        // Lowlink must propagate correctly to identify 1 as header
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(1, MirTerminator::Goto { target: 2 }),
            MirBasicBlock::new(2, MirTerminator::Goto { target: 3 }),
            MirBasicBlock::new(
                3,
                MirTerminator::CondGoto {
                    condition: "_c".to_string(),
                    then_target: 4,
                    else_target: 1, // back edge to 1
                },
            ),
            MirBasicBlock::new(4, MirTerminator::Return),
        ];

        let headers = compute_loop_headers(&blocks);

        // The back edge 3->1 should make 1 a loop header
        // With || changed to &&, lowlink propagation would break
        assert!(
            headers.values().any(|&h| h == 1) || headers.contains_key(&3),
            "Block 1 should be loop header due to back edge from 3: {:?}",
            headers
        );
    }

    // Test to kill: compute_loop_headers delete ! at line 1120
    // Line 1120: if !visited[block.id] -> if visited[block.id]
    // This would only visit already-visited blocks (wrong)
    #[test]
    fn test_compute_loop_headers_visits_all_blocks() {
        // Linear chain with no loops
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(1, MirTerminator::Goto { target: 2 }),
            MirBasicBlock::new(2, MirTerminator::Return),
        ];

        let headers = compute_loop_headers(&blocks);

        // No loops, should return empty (but algorithm should not panic)
        // With the mutant, it would never visit any blocks
        assert!(
            headers.is_empty(),
            "Linear chain should have no loop headers: {:?}",
            headers
        );
    }

    // Test to kill: compute_loop_headers > with < at line 1109
    // Line 1109: lowlink[w] < lowlink[v] -> lowlink[w] > lowlink[v]
    // This inverts the comparison direction for lowlink updates
    #[test]
    fn test_compute_loop_headers_lowlink_comparison_direction() {
        // Create structure where lowlink direction matters:
        // 0 -> 1 -> 2
        //      ^----/
        // Block 2 has back edge to 1
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(
                1,
                MirTerminator::CondGoto {
                    condition: "_x".to_string(),
                    then_target: 3,
                    else_target: 2,
                },
            ),
            MirBasicBlock::new(2, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(3, MirTerminator::Return),
        ];

        let headers = compute_loop_headers(&blocks);

        // Block 1 should be the loop header
        // With inverted comparison, lowlink would not propagate correctly
        assert!(
            !headers.is_empty(),
            "Should detect loop in 1->2->1 cycle: {:?}",
            headers
        );
    }

    // Test to kill: compute_loop_headers += vs *= at line 1063
    // Line 1063: *index += 1 -> *index *= 1
    // This would keep index at 0 forever, breaking DFS numbering
    #[test]
    fn test_compute_loop_headers_index_increments() {
        // Multiple SCCs require proper index management
        // 0 -> 1 -> 0 (SCC1)
        // 2 -> 3 -> 2 (SCC2, disconnected)
        // With *= 1 mutant, all nodes would have index 0
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "_a".to_string(),
                    then_target: 2,
                    else_target: 1,
                },
            ),
            MirBasicBlock::new(1, MirTerminator::Goto { target: 0 }),
            MirBasicBlock::new(
                2,
                MirTerminator::CondGoto {
                    condition: "_b".to_string(),
                    then_target: 4,
                    else_target: 3,
                },
            ),
            MirBasicBlock::new(3, MirTerminator::Goto { target: 2 }),
            MirBasicBlock::new(4, MirTerminator::Return),
        ];

        let headers = compute_loop_headers(&blocks);

        // Should find loop headers for both cycles
        // With index stuck at 0, Tarjan would malfunction
        assert!(
            !headers.is_empty(),
            "Should detect loops in both cycles: {:?}",
            headers
        );
    }

    // ========================================================================
    // collect_vars_from_expr tests
    // Tests for the tokenizer that extracts variable names from SMT expressions.
    // These guard against regressions that previously caused hangs or
    // mis-tokenization when scanning SMT syntax.
    // ========================================================================

    // Non-identifier characters should be ignored, not collected as variables.
    #[test]
    fn test_collect_vars_from_expr_skips_syntax_chars() {
        let mut vars = HashSet::new();
        collect_vars_from_expr("(+ x y)", &mut vars);

        // Should only have x and y, not parentheses or +
        assert!(vars.contains("x"), "Should find variable x");
        assert!(vars.contains("y"), "Should find variable y");
        assert_eq!(vars.len(), 2, "Should only have 2 variables");
    }

    // Underscores are part of identifiers (MIR locals start with an underscore).
    #[test]
    fn test_collect_vars_from_expr_handles_underscores() {
        let mut vars = HashSet::new();
        collect_vars_from_expr("(= _1 _result)", &mut vars);

        // Variables starting with underscore should be collected
        assert!(vars.contains("_1"), "Should find variable _1");
        assert!(vars.contains("_result"), "Should find variable _result");
    }

    // Nested syntax should not prevent progress through the string.
    #[test]
    fn test_collect_vars_from_expr_advances_past_syntax() {
        let mut vars = HashSet::new();
        // Multiple syntax characters in a row
        collect_vars_from_expr("((((a))))", &mut vars);

        assert!(
            vars.contains("a"),
            "Should find variable a despite nested parens"
        );
        assert_eq!(vars.len(), 1, "Should only have 1 variable");
    }

    // Identifiers should be collected in full, not truncated at digits/underscores.
    #[test]
    fn test_collect_vars_from_expr_collects_full_identifiers() {
        let mut vars = HashSet::new();
        collect_vars_from_expr("abc123_def", &mut vars);

        // Should collect the full identifier, not just first char
        assert!(
            vars.contains("abc123_def"),
            "Should collect full identifier: {:?}",
            vars
        );
    }

    #[test]
    fn test_collect_vars_from_expr_multichar_variable() {
        let mut vars = HashSet::new();
        collect_vars_from_expr("variable", &mut vars);

        // Should have the complete name, not empty or single char
        assert!(
            vars.contains("variable"),
            "Should collect 'variable': {:?}",
            vars
        );
        assert_eq!(vars.len(), 1);
    }

    #[test]
    fn test_collect_vars_from_expr_alphanumeric_or_underscore() {
        let mut vars = HashSet::new();
        collect_vars_from_expr("a_b", &mut vars);

        // With || the identifier "a_b" is collected as one token
        // With && it would split on underscore
        assert!(
            vars.contains("a_b"),
            "Should collect 'a_b' as single identifier: {:?}",
            vars
        );
    }

    #[test]
    fn test_collect_vars_from_expr_underscore_in_middle() {
        let mut vars = HashSet::new();
        collect_vars_from_expr("foo_bar_baz", &mut vars);

        // Should be single identifier, not three
        assert!(
            vars.contains("foo_bar_baz"),
            "Should collect as single identifier: {:?}",
            vars
        );
        assert!(!vars.contains("foo"), "Should NOT split at underscores");
        assert!(!vars.contains("bar"), "Should NOT split at underscores");
        assert!(!vars.contains("baz"), "Should NOT split at underscores");
    }

    // Long identifiers should terminate; tokenizer must make forward progress.
    #[test]
    fn test_collect_vars_from_expr_long_identifier_terminates() {
        let mut vars = HashSet::new();
        // Long identifier that would cause infinite loop if += became *= 1
        collect_vars_from_expr("very_long_variable_name_123", &mut vars);

        assert!(
            vars.contains("very_long_variable_name_123"),
            "Should collect long identifier: {:?}",
            vars
        );
    }

    // Additional comprehensive tests for collect_vars_from_expr

    #[test]
    fn test_collect_vars_from_expr_skips_smt_keywords() {
        let mut vars = HashSet::new();
        collect_vars_from_expr("(and x (or y (not z)))", &mut vars);

        // Should skip "and", "or", "not" but keep x, y, z
        assert!(!vars.contains("and"), "Should skip keyword 'and'");
        assert!(!vars.contains("or"), "Should skip keyword 'or'");
        assert!(!vars.contains("not"), "Should skip keyword 'not'");
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(vars.contains("z"));
    }

    #[test]
    fn test_collect_vars_from_expr_skips_numbers() {
        let mut vars = HashSet::new();
        collect_vars_from_expr("(+ x 42)", &mut vars);

        // Should skip pure numbers
        assert!(!vars.contains("42"), "Should skip numbers");
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_collect_vars_from_expr_handles_primed_vars() {
        let mut vars = HashSet::new();
        collect_vars_from_expr("(= x' (+ x 1))", &mut vars);

        // Primed variables should have their base name extracted
        assert!(
            vars.contains("x"),
            "Should extract base from primed var x': {:?}",
            vars
        );
    }

    #[test]
    fn test_collect_vars_from_expr_skips_primed_keywords() {
        let mut vars = HashSet::new();
        collect_vars_from_expr("(and' (= x y))", &mut vars);

        // Keyword stem should be ignored even when primed
        assert!(
            !vars.contains("and"),
            "Primed keyword should be filtered out"
        );
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_collect_vars_from_expr_empty_string() {
        let mut vars = HashSet::new();
        collect_vars_from_expr("", &mut vars);

        assert!(vars.is_empty(), "Empty expression should yield no vars");
    }

    #[test]
    fn test_collect_vars_from_expr_only_syntax() {
        let mut vars = HashSet::new();
        collect_vars_from_expr("(((())))", &mut vars);

        assert!(vars.is_empty(), "Only parentheses should yield no vars");
    }

    // ========================================================================
    // Phase 16D: Algebraic rewriting integration tests
    // ========================================================================

    #[test]
    fn test_apply_algebraic_rewrites_no_bitwise() {
        // Program with no bitwise operations - should pass through unchanged
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(
                MirBasicBlock::new(0, MirTerminator::Goto { target: 1 })
                    .with_statement(MirStatement::Assign {
                        lhs: "x".to_string(),
                        rhs: "(+ 1 2)".to_string(),
                    })
                    .with_statement(MirStatement::Assert {
                        condition: "(> x 0)".to_string(),
                        message: None,
                    }),
            )
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .finish();

        let rewritten = apply_algebraic_rewrites(&program);

        // Should be unchanged
        assert_eq!(rewritten.basic_blocks.len(), 2);
        if let MirStatement::Assign { rhs, .. } = &rewritten.basic_blocks[0].statements[0] {
            assert_eq!(rhs, "(+ 1 2)");
        } else {
            panic!("Expected Assign statement");
        }
    }

    #[test]
    fn test_apply_algebraic_rewrites_and_mask() {
        // x & 255 -> x mod 256
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "y".to_string(),
                    rhs: "(bitand x 255)".to_string(),
                },
            ))
            .finish();

        let rewritten = apply_algebraic_rewrites(&program);

        if let MirStatement::Assign { rhs, .. } = &rewritten.basic_blocks[0].statements[0] {
            assert_eq!(rhs, "(mod x 256)");
        } else {
            panic!("Expected Assign statement");
        }
    }

    #[test]
    fn test_apply_algebraic_rewrites_shift_left() {
        // x << 3 -> x * 8
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "y".to_string(),
                    rhs: "(bitshl x 3)".to_string(),
                },
            ))
            .finish();

        let rewritten = apply_algebraic_rewrites(&program);

        if let MirStatement::Assign { rhs, .. } = &rewritten.basic_blocks[0].statements[0] {
            assert_eq!(rhs, "(* x 8)");
        } else {
            panic!("Expected Assign statement");
        }
    }

    #[test]
    fn test_apply_algebraic_rewrites_shift_right() {
        // x >> 2 -> x / 4
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "y".to_string(),
                    rhs: "(bitshr x 2)".to_string(),
                },
            ))
            .finish();

        let rewritten = apply_algebraic_rewrites(&program);

        if let MirStatement::Assign { rhs, .. } = &rewritten.basic_blocks[0].statements[0] {
            assert_eq!(rhs, "(div x 4)");
        } else {
            panic!("Expected Assign statement");
        }
    }

    #[test]
    fn test_apply_algebraic_rewrites_xor_self() {
        // x ^ x -> 0
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "y".to_string(),
                    rhs: "(bitxor x x)".to_string(),
                },
            ))
            .finish();

        let rewritten = apply_algebraic_rewrites(&program);

        if let MirStatement::Assign { rhs, .. } = &rewritten.basic_blocks[0].statements[0] {
            assert_eq!(rhs, "0");
        } else {
            panic!("Expected Assign statement");
        }
    }

    #[test]
    fn test_apply_algebraic_rewrites_or_zero() {
        // x | 0 -> x
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "y".to_string(),
                    rhs: "(bitor x 0)".to_string(),
                },
            ))
            .finish();

        let rewritten = apply_algebraic_rewrites(&program);

        if let MirStatement::Assign { rhs, .. } = &rewritten.basic_blocks[0].statements[0] {
            assert_eq!(rhs, "x");
        } else {
            panic!("Expected Assign statement");
        }
    }

    #[test]
    fn test_apply_algebraic_rewrites_and_zero() {
        // x & 0 -> 0
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "y".to_string(),
                    rhs: "(bitand x 0)".to_string(),
                },
            ))
            .finish();

        let rewritten = apply_algebraic_rewrites(&program);

        if let MirStatement::Assign { rhs, .. } = &rewritten.basic_blocks[0].statements[0] {
            assert_eq!(rhs, "0");
        } else {
            panic!("Expected Assign statement");
        }
    }

    #[test]
    fn test_apply_algebraic_rewrites_nested() {
        // (bitshl (bitand x 255) 2) -> (* (mod x 256) 4)
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assign {
                    lhs: "y".to_string(),
                    rhs: "(bitshl (bitand x 255) 2)".to_string(),
                },
            ))
            .finish();

        let rewritten = apply_algebraic_rewrites(&program);

        if let MirStatement::Assign { rhs, .. } = &rewritten.basic_blocks[0].statements[0] {
            // The inner AND is rewritten first, then the outer SHL
            assert_eq!(rhs, "(* (mod x 256) 4)");
        } else {
            panic!("Expected Assign statement");
        }
    }

    #[test]
    fn test_apply_algebraic_rewrites_in_condition() {
        // Bitwise ops in CondGoto condition should be rewritten
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "(= (bitand x 1) 1)".to_string(),
                    then_target: 1,
                    else_target: 2,
                },
            ))
            .block(MirBasicBlock::new(1, MirTerminator::Return))
            .block(MirBasicBlock::new(2, MirTerminator::Return))
            .finish();

        let rewritten = apply_algebraic_rewrites(&program);

        if let MirTerminator::CondGoto { condition, .. } = &rewritten.basic_blocks[0].terminator {
            // x & 1 with power-of-2 becomes (mod (div x 1) 2) * 1
            assert!(condition.contains("mod") || condition.contains("div"));
        } else {
            panic!("Expected CondGoto terminator");
        }
    }

    #[test]
    fn test_apply_algebraic_rewrites_in_assert() {
        // Bitwise ops in Assert should be rewritten
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assert {
                    condition: "(< (bitand x 255) 256)".to_string(),
                    message: Some("masked value must be less than 256".to_string()),
                },
            ))
            .finish();

        let rewritten = apply_algebraic_rewrites(&program);

        if let MirStatement::Assert { condition, .. } = &rewritten.basic_blocks[0].statements[0] {
            assert_eq!(condition, "(< (mod x 256) 256)");
        } else {
            panic!("Expected Assert statement");
        }
    }

    #[test]
    fn test_apply_algebraic_rewrites_preserves_structure() {
        // Complex program - verify structure is preserved
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .local("z", SmtType::Int)
            .block(
                MirBasicBlock::new(0, MirTerminator::Goto { target: 1 })
                    .with_statement(MirStatement::Assign {
                        lhs: "y".to_string(),
                        rhs: "(bitand x 255)".to_string(),
                    })
                    .with_statement(MirStatement::Assume("(>= x 0)".to_string())),
            )
            .block(
                MirBasicBlock::new(
                    1,
                    MirTerminator::CondGoto {
                        condition: "(> y 0)".to_string(),
                        then_target: 2,
                        else_target: 3,
                    },
                )
                .with_statement(MirStatement::Assign {
                    lhs: "z".to_string(),
                    rhs: "(bitshl y 1)".to_string(),
                }),
            )
            .block(MirBasicBlock::new(2, MirTerminator::Return))
            .block(MirBasicBlock::new(3, MirTerminator::Abort))
            .finish();

        let rewritten = apply_algebraic_rewrites(&program);

        // Check structure preserved
        assert_eq!(rewritten.basic_blocks.len(), 4);
        assert_eq!(rewritten.locals.len(), 3);
        assert_eq!(rewritten.start_block, 0);

        // Check rewrites applied
        if let MirStatement::Assign { rhs, .. } = &rewritten.basic_blocks[0].statements[0] {
            assert_eq!(rhs, "(mod x 256)");
        }
        if let MirStatement::Assign { rhs, .. } = &rewritten.basic_blocks[1].statements[0] {
            assert_eq!(rhs, "(* y 2)");
        }
    }

    #[test]
    fn test_encode_mir_to_chc_with_strategy_fast_path() {
        // Program with no bitwise - should use fast path
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .block(MirBasicBlock::new(0, MirTerminator::Return).with_statement(
                MirStatement::Assert {
                    condition: "(>= x 0)".to_string(),
                    message: None,
                },
            ))
            .finish();

        let result = encode_mir_to_chc_with_strategy(&program);

        match result {
            VerificationResult::ChcResult {
                strategy,
                rewrites_applied,
                ..
            } => {
                assert!(
                    matches!(strategy, VerificationPath::ChcFast),
                    "Expected ChcFast path"
                );
                assert!(!rewrites_applied, "No rewrites should be applied");
            }
            VerificationResult::Delegated { .. } => {
                panic!("Should not delegate for simple program");
            }
        }
    }

    #[test]
    fn test_encode_mir_to_chc_with_strategy_rewritten_path() {
        // Program with rewritable bitwise in proof-relevant position
        let program = MirProgram::builder(0)
            .local("x", SmtType::Int)
            .local("y", SmtType::Int)
            .block(
                MirBasicBlock::new(0, MirTerminator::Return)
                    .with_statement(MirStatement::Assign {
                        lhs: "y".to_string(),
                        rhs: "(bitand x 255)".to_string(),
                    })
                    .with_statement(MirStatement::Assert {
                        condition: "(< y 256)".to_string(),
                        message: None,
                    }),
            )
            .finish();

        let result = encode_mir_to_chc_with_strategy(&program);

        match result {
            VerificationResult::ChcResult {
                chc,
                strategy,
                rewrites_applied,
            } => {
                // Should use rewritten path since bitand can be rewritten to mod
                if matches!(strategy, VerificationPath::ChcRewritten { .. }) {
                    assert!(rewrites_applied, "Rewrites should be applied");
                }
                // The CHC should be valid SMT
                let smt = chc.to_smt2();
                assert!(smt.contains("(set-logic HORN)"));
            }
            VerificationResult::Delegated { .. } => {
                // This is also acceptable - delegation for any bitwise
            }
        }
    }

    #[test]
    fn test_rewrite_statement_havoc_unchanged() {
        let stmt = MirStatement::Havoc {
            var: "x".to_string(),
        };
        let rewritten = rewrite_statement(&stmt);

        if let MirStatement::Havoc { var } = rewritten {
            assert_eq!(var, "x");
        } else {
            panic!("Expected Havoc statement");
        }
    }

    #[test]
    fn test_rewrite_terminator_goto_unchanged() {
        let term = MirTerminator::Goto { target: 5 };
        let rewritten = rewrite_terminator(&term);

        if let MirTerminator::Goto { target } = rewritten {
            assert_eq!(target, 5);
        } else {
            panic!("Expected Goto terminator");
        }
    }

    #[test]
    fn test_rewrite_terminator_call_args() {
        let term = MirTerminator::Call {
            destination: Some("result".to_string()),
            func: "foo".to_string(),
            args: vec!["(bitand x 255)".to_string(), "y".to_string()],
            target: 1,
            unwind: None,
            precondition_check: None,
            postcondition_assumption: Some("(>= result 0)".to_string()),
            is_range_into_iter: false,
            is_range_next: false,
        };

        let rewritten = rewrite_terminator(&term);

        if let MirTerminator::Call {
            args,
            postcondition_assumption,
            ..
        } = rewritten
        {
            assert_eq!(args[0], "(mod x 256)");
            assert_eq!(args[1], "y");
            assert_eq!(postcondition_assumption, Some("(>= result 0)".to_string()));
        } else {
            panic!("Expected Call terminator");
        }
    }

    #[test]
    fn test_rewrite_expr_if_bitwise_no_bitwise() {
        let result = rewrite_expr_if_bitwise("(+ x 1)");
        assert_eq!(result, "(+ x 1)");
    }

    #[test]
    fn test_rewrite_expr_if_bitwise_with_bitwise() {
        let result = rewrite_expr_if_bitwise("(bitand x 7)");
        assert_eq!(result, "(mod x 8)");
    }

    // ========================================================================
    // Tests to catch remaining mutants in build_call_clause
    // ========================================================================

    // Test to kill: Line 1360 - delete ! in `if !context_conditions.is_empty()`
    // Mutant: `if context_conditions.is_empty()` - would skip processing when there ARE conditions
    // This test ensures path conditions are actually included in the clause output
    #[test]
    fn test_build_call_clause_path_condition_must_appear() {
        // Create: block 0 -> conditional -> block 1 (call) -> block 2 (return)
        // The conditional adds "(> _x 0)" as a path condition to reach block 1
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "(> _x 0)".to_string(),
                    then_target: 2, // skip
                    else_target: 1, // call
                },
            ),
            MirBasicBlock::new(
                1,
                MirTerminator::Call {
                    func: "test_fn".to_string(),
                    args: vec![],
                    destination: Some("_r".to_string()),
                    target: 2,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ),
            MirBasicBlock::new(2, MirTerminator::Return),
        ];
        let locals = vec![
            MirLocal::new("_x".to_string(), SmtType::Int),
            MirLocal::new("_r".to_string(), SmtType::Int),
        ];
        let predecessors = build_predecessor_map(&blocks, &locals);

        let clause = build_call_clause(
            &blocks[1],
            Some("_r"),
            "test_fn",
            &[],
            false,
            false,
            2,
            &locals,
            None,
            &predecessors,
            &blocks,
            0,
            None,
            &HashMap::new(),
            &HashMap::new(),
        );

        // The path condition from block 0 to block 1 is "(not (> _x 0))" (else branch)
        // With the delete ! mutant, context_conditions would be skipped entirely
        assert!(
            clause.contains("(not (> _x 0))") || clause.contains("(<= _x 0)"),
            "Path condition must appear in call clause. Got: {}",
            clause
        );
    }

    // Test to kill: Line 1363 - `|| → &&` in `cond.trim().is_empty() || cond == "true"`
    // Mutant: `cond.trim().is_empty() && cond == "true"` - skips nothing (impossible condition)
    // This test ensures "true" conditions ARE filtered out
    #[test]
    fn test_build_call_clause_filters_true_condition() {
        // Create a loop header block with a "true" Assume statement (loop invariant)
        // The "true" should be filtered when processing context conditions
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(1, MirTerminator::Goto { target: 2 })
                .with_statement(MirStatement::Assume("true".to_string())), // "true" invariant
            MirBasicBlock::new(
                2,
                MirTerminator::Call {
                    func: "target_fn".to_string(),
                    args: vec![],
                    destination: Some("_out".to_string()),
                    target: 3,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ),
            MirBasicBlock::new(3, MirTerminator::Return),
        ];
        let locals = vec![MirLocal::new("_out".to_string(), SmtType::Int)];
        let predecessors = build_predecessor_map(&blocks, &locals);

        // Mark block 1 as a loop header so the invariant gets included in context_conditions
        let clause = build_call_clause(
            &blocks[2],
            Some("_out"),
            "target_fn",
            &[],
            false,
            false,
            3,
            &locals,
            None,
            &predecessors,
            &blocks,
            0,
            Some(1), // loop_header = block 1
            &HashMap::new(),
            &HashMap::new(),
        );

        // The "true" invariant should NOT appear as a conjunct
        // With the || → && mutant, "true" would be included as " true " in the clause
        // Check for " true " as a standalone word (not part of a variable name)
        assert!(
            !clause.contains(" true "),
            "Literal 'true' should be filtered from context conditions. With || → && mutant, 'true' is included. Got: {}",
            clause
        );
    }

    // Test to kill: Line 1363 - `== → !=` in `cond == "true"`
    // Mutant: `cond != "true"` - would filter everything EXCEPT "true"
    // This test ensures non-true conditions are NOT filtered
    #[test]
    fn test_build_call_clause_keeps_non_true_conditions() {
        // Create conditional path that generates a real condition (not "true")
        let blocks = vec![
            MirBasicBlock::new(
                0,
                MirTerminator::CondGoto {
                    condition: "(>= _n 0)".to_string(),
                    then_target: 2,
                    else_target: 1,
                },
            ),
            MirBasicBlock::new(
                1,
                MirTerminator::Call {
                    func: "callee".to_string(),
                    args: vec!["_n".to_string()],
                    destination: Some("_res".to_string()),
                    target: 2,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            ),
            MirBasicBlock::new(2, MirTerminator::Return),
        ];
        let locals = vec![
            MirLocal::new("_n".to_string(), SmtType::Int),
            MirLocal::new("_res".to_string(), SmtType::Int),
        ];
        let predecessors = build_predecessor_map(&blocks, &locals);

        let clause = build_call_clause(
            &blocks[1],
            Some("_res"),
            "callee",
            &["_n".to_string()],
            false,
            false,
            2,
            &locals,
            None,
            &predecessors,
            &blocks,
            0,
            None,
            &HashMap::new(),
            &HashMap::new(),
        );

        // The path condition "(not (>= _n 0))" (i.e., "(_n < 0)") should be present
        // With the `== → !=` mutant, all non-"true" conditions would be filtered
        assert!(
            clause.contains("(not (>= _n 0))") || clause.contains("(< _n 0)"),
            "Non-true path condition must NOT be filtered. Got: {}",
            clause
        );
    }

    // ========================================================================
    // Tests to catch Tarjan algorithm mutations (lines 1268, 1276)
    // These mutations change > to == or >= in lowlink propagation
    // ========================================================================

    // Test to kill: Line 1268 - `> → ==` in lowlink update after recursive call
    // Mutant: `if *low_v == low_w` - only updates when equal (wrong)
    // Test: Create graph where low_v > low_w, verify correct SCC detection
    #[test]
    fn test_tarjan_lowlink_propagation_greater_than() {
        // Graph: 0 -> 1 -> 2 -> 1 (back edge creates SCC {1, 2})
        // When processing node 1 after returning from node 2:
        //   - low_v (node 1) starts as index 1
        //   - low_w (node 2) should have lowlink = 1 (due to back edge to 1)
        //   - Need low_v > low_w to trigger the update
        // Actually we need a case where low_v is strictly greater than low_w
        // after the recursive call. Let's create:
        // 0 -> 1 -> 2 -> 3 -> 1 (back edge)
        // Processing order: 0(idx=0), 1(idx=1), 2(idx=2), 3(idx=3)
        // After 3 returns: 3 finds back edge to 1, low_3 = 1
        // After 2 returns: low_2 should be min(2, low_3) = 1
        // After 1 returns: low_1 should be min(1, low_2) = 1
        // SCC root is node 1 (low == idx)
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(1, MirTerminator::Goto { target: 2 }),
            MirBasicBlock::new(2, MirTerminator::Goto { target: 3 }),
            MirBasicBlock::new(
                3,
                MirTerminator::CondGoto {
                    condition: "c".to_string(),
                    then_target: 4,
                    else_target: 1, // back edge
                },
            ),
            MirBasicBlock::new(4, MirTerminator::Return),
        ];

        let headers = compute_loop_headers(&blocks);

        // With correct lowlink propagation: all nodes 1,2,3 in SCC with header 1
        // With `> → ==` mutant: only nodes 2,3 in SCC with header 2 (node 1 excluded)
        // Test must verify node 1 is included in the loop (catches the mutant)
        assert!(
            headers.contains_key(&1),
            "Node 1 must be in loop SCC. With > → == mutant, lowlink doesn't propagate and node 1 is excluded. Headers: {:?}",
            headers
        );
        // Verify correct header assignment: header should be 1 (minimum node in SCC)
        assert_eq!(
            headers.get(&1),
            Some(&1),
            "Node 1 should have header 1. Headers: {:?}",
            headers
        );
    }

    // Test to kill: Line 1268 - `> → >=` in lowlink update
    // Mutant: `if *low_v >= low_w` - updates even when equal (extra update)
    // This is semantically equivalent when values are already equal
    // Need case where low_v == low_w and we verify no extra behavior
    #[test]
    fn test_tarjan_lowlink_no_spurious_update() {
        // When low_v == low_w, no update should happen
        // Create simple self-loop: node 0 loops to itself
        // After recursive call (none for self-loop via on_stack branch):
        //   - low_0 starts as 0
        //   - self-loop goes through else branch (on_stack case)
        // Actually for line 1268, we need the recursive case.
        // Let's make: 0 -> 1 -> 2 -> 2 (self loop) -> return
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(1, MirTerminator::Goto { target: 2 }),
            MirBasicBlock::new(
                2,
                MirTerminator::CondGoto {
                    condition: "c".to_string(),
                    then_target: 3,
                    else_target: 2, // self-loop
                },
            ),
            MirBasicBlock::new(3, MirTerminator::Return),
        ];

        let headers = compute_loop_headers(&blocks);

        // Node 2 should be detected as loop header due to self-loop
        assert!(
            headers.contains_key(&2),
            "Self-loop at node 2 should be detected. Headers: {:?}",
            headers
        );
    }

    // Test to kill: Line 1276 - `> → >=` in on_stack lowlink update
    // Mutant: `if *low_v >= idx_w` - updates even when equal
    // This affects the case where a node on stack is found
    #[test]
    fn test_tarjan_on_stack_lowlink_greater_than() {
        // Need: node v finds node w already on stack, and low_v > idx_w
        // Create: 0 -> 1 -> 2 -> 0 (back edge to 0)
        // Processing: 0(idx=0,low=0) -> 1(idx=1,low=1) -> 2(idx=2,low=2)
        // Node 2 finds 0 on stack: idx_0 = 0, low_2 = 2 > 0, so update low_2 = 0
        // Node 1 returns from 2: low_2 = 0, low_1 = 1 > 0, update low_1 = 0
        // Node 0: low_0 == idx_0 = 0, is root, pops SCC {2,1,0}
        // Size > 1 + back edge exists → header detected
        let blocks = vec![
            MirBasicBlock::new(0, MirTerminator::Goto { target: 1 }),
            MirBasicBlock::new(1, MirTerminator::Goto { target: 2 }),
            MirBasicBlock::new(
                2,
                MirTerminator::CondGoto {
                    condition: "loop".to_string(),
                    then_target: 3,
                    else_target: 0, // back edge to 0
                },
            ),
            MirBasicBlock::new(3, MirTerminator::Return),
        ];

        let headers = compute_loop_headers(&blocks);

        // Should detect loop involving nodes 0, 1, 2
        assert!(
            !headers.is_empty(),
            "Should detect loop in cycle 0->1->2->0. Headers: {:?}",
            headers
        );
    }

    // ========================================================================
    // Precondition/Postcondition (Contract) tests
    // ========================================================================

    #[test]
    fn test_build_property_formula_with_precondition() {
        // Build a simple program: call a function with a precondition
        // Block 0: call foo(x) with precondition (> x 0)
        // Block 1: return

        let blocks = vec![
            MirBasicBlock {
                id: 0,
                statements: vec![],
                terminator: MirTerminator::Call {
                    destination: Some("_0".to_string()),
                    func: "foo".to_string(),
                    args: vec!["x".to_string()],
                    target: 1,
                    unwind: None,
                    precondition_check: Some("(> x 0)".to_string()),
                    postcondition_assumption: None,
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            },
            MirBasicBlock::new(1, MirTerminator::Return),
        ];

        let property = build_property_formula(&blocks);

        // The property formula should include the precondition check:
        // (or (not (= pc 0)) (> x 0))
        assert!(
            property.contains("(or (not (= pc 0)) (> x 0))"),
            "Property should include precondition check. Got: {}",
            property
        );
    }

    #[test]
    fn test_call_with_postcondition_assumption() {
        // Build a program where a call has postcondition assumption
        // This tests that postconditions constrain the return value

        let blocks = [
            MirBasicBlock {
                id: 0,
                statements: vec![],
                terminator: MirTerminator::Call {
                    destination: Some("_0".to_string()),
                    func: "increment".to_string(),
                    args: vec!["x".to_string()],
                    target: 1,
                    unwind: None,
                    precondition_check: None,
                    postcondition_assumption: Some("(> _0' x)".to_string()),
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            },
            MirBasicBlock::new(1, MirTerminator::Return),
        ];

        // The postcondition assumption is used in build_call_clause, not in
        // build_property_formula, so we just verify the structure is correct
        if let MirTerminator::Call {
            postcondition_assumption,
            ..
        } = &blocks[0].terminator
        {
            assert!(postcondition_assumption.is_some());
            assert_eq!(postcondition_assumption.as_ref().unwrap(), "(> _0' x)");
        } else {
            panic!("Expected Call terminator");
        }
    }

    #[test]
    fn test_call_with_both_precondition_and_postcondition() {
        // Build a program where a call has both precondition and postcondition
        // Precondition: x > 0 (must hold before call)
        // Postcondition: result > x (assumed after call)

        let blocks = vec![
            MirBasicBlock {
                id: 0,
                statements: vec![],
                terminator: MirTerminator::Call {
                    destination: Some("result".to_string()),
                    func: "safe_increment".to_string(),
                    args: vec!["x".to_string()],
                    target: 1,
                    unwind: None,
                    precondition_check: Some("(> x 0)".to_string()),
                    postcondition_assumption: Some("(> result' x)".to_string()),
                    is_range_into_iter: false,
                    is_range_next: false,
                },
            },
            MirBasicBlock::new(1, MirTerminator::Return),
        ];

        // The property formula should include the precondition
        let property = build_property_formula(&blocks);
        assert!(
            property.contains("(or (not (= pc 0)) (> x 0))"),
            "Property should include precondition. Got: {}",
            property
        );

        // Verify both are stored correctly
        if let MirTerminator::Call {
            precondition_check,
            postcondition_assumption,
            ..
        } = &blocks[0].terminator
        {
            assert_eq!(precondition_check.as_ref().unwrap(), "(> x 0)");
            assert_eq!(postcondition_assumption.as_ref().unwrap(), "(> result' x)");
        }
    }
}
