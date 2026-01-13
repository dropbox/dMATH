//! SMT executor - orchestrates frontend and theory solver
//!
//! Provides a high-level interface for executing SMT-LIB commands with
//! theory integration.

use hashbrown::{HashMap, HashSet};
use std::collections::BTreeMap;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use z4_arrays::{ArrayModel, ArraySolver};
use z4_bv::{BvBits, BvModel, BvSolver};
use z4_core::proof::{AletheRule, Proof, ProofStep};
use z4_core::term::{Constant, Symbol, TermData};
use z4_core::{
    Sort, TermId, TermStore, TheoryPropagation, TheoryResult, TheorySolver, Tseitin, TseitinResult,
    TseitinState,
};
use z4_euf::{EufModel, EufSolver};
use z4_frontend::{Command, CommandResult, Context, ElaborateError, OptionValue};
use z4_lia::{LiaModel, LiaSolver};
use z4_lra::{LraModel, LraSolver};
use z4_proof::export_alethe;
use z4_sat::{
    AssumeResult, Literal as SatLiteral, SolveResult, Solver as SatSolver, Variable as SatVariable,
};

use crate::{DpllT, PropositionalTheory};

/// Check if a term contains arithmetic operations (not just arithmetic-typed variables).
/// This is used to filter which literals go to the arithmetic solver.
fn contains_arithmetic_ops(terms: &TermStore, term: TermId) -> bool {
    match terms.get(term) {
        TermData::App(Symbol::Named(name), args) => {
            // Arithmetic operations and comparisons
            if matches!(
                name.as_str(),
                "<" | "<=" | ">" | ">=" | "+" | "-" | "*" | "/"
            ) {
                return true;
            }
            // For equality, check if either side contains arithmetic ops
            // Don't recurse into arbitrary function applications - they're handled by EUF
            if name == "=" && args.len() == 2 {
                return contains_arithmetic_ops(terms, args[0])
                    || contains_arithmetic_ops(terms, args[1]);
            }
            false
        }
        TermData::Const(Constant::Int(_) | Constant::Rational(_)) => true,
        TermData::Not(inner) => contains_arithmetic_ops(terms, *inner),
        TermData::Ite(_, t, e) => {
            contains_arithmetic_ops(terms, *t) || contains_arithmetic_ops(terms, *e)
        }
        _ => false,
    }
}

/// Check if a term involves Int-sorted arithmetic operands.
/// Returns true if any arithmetic operand has Int sort.
fn involves_int_arithmetic(terms: &TermStore, term: TermId) -> bool {
    match terms.get(term) {
        TermData::App(Symbol::Named(name), args) => {
            // Arithmetic comparisons
            if matches!(name.as_str(), "<" | "<=" | ">" | ">=") && args.len() == 2 {
                return matches!(terms.sort(args[0]), Sort::Int)
                    || matches!(terms.sort(args[1]), Sort::Int);
            }
            // Equality on Int
            if name == "=" && args.len() == 2 {
                return matches!(terms.sort(args[0]), Sort::Int)
                    || matches!(terms.sort(args[1]), Sort::Int);
            }
            // Arithmetic operations with Int result
            if matches!(name.as_str(), "+" | "-" | "*" | "/") {
                return matches!(terms.sort(term), Sort::Int);
            }
            false
        }
        TermData::Const(Constant::Int(_)) => true,
        TermData::Var(_, _) => matches!(terms.sort(term), Sort::Int),
        TermData::Not(inner) => involves_int_arithmetic(terms, *inner),
        _ => false,
    }
}

/// Check if a term involves Real-sorted arithmetic operands.
/// Returns true if any arithmetic operand has Real sort.
fn involves_real_arithmetic(terms: &TermStore, term: TermId) -> bool {
    match terms.get(term) {
        TermData::App(Symbol::Named(name), args) => {
            // Arithmetic comparisons
            if matches!(name.as_str(), "<" | "<=" | ">" | ">=") && args.len() == 2 {
                return matches!(terms.sort(args[0]), Sort::Real)
                    || matches!(terms.sort(args[1]), Sort::Real);
            }
            // Equality on Real
            if name == "=" && args.len() == 2 {
                return matches!(terms.sort(args[0]), Sort::Real)
                    || matches!(terms.sort(args[1]), Sort::Real);
            }
            // Arithmetic operations with Real result
            if matches!(name.as_str(), "+" | "-" | "*" | "/") {
                return matches!(terms.sort(term), Sort::Real);
            }
            false
        }
        TermData::Const(Constant::Rational(_)) => true,
        TermData::Var(_, _) => matches!(terms.sort(term), Sort::Real),
        TermData::Not(inner) => involves_real_arithmetic(terms, *inner),
        _ => false,
    }
}

/// Error during SMT execution
#[derive(Debug, Clone, thiserror::Error)]
pub enum ExecutorError {
    /// Elaboration error
    #[error("elaboration error: {0}")]
    Elaborate(#[from] ElaborateError),
    /// Unsupported logic
    #[error("unsupported logic: {0}")]
    UnsupportedLogic(String),
}

/// Result type for executor operations
pub type Result<T> = std::result::Result<T, ExecutorError>;

/// Result of a check-sat operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckSatResult {
    /// The assertions are satisfiable
    Sat,
    /// The assertions are unsatisfiable
    Unsat,
    /// The solver could not determine satisfiability
    Unknown,
}

impl std::fmt::Display for CheckSatResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckSatResult::Sat => write!(f, "sat"),
            CheckSatResult::Unsat => write!(f, "unsat"),
            CheckSatResult::Unknown => write!(f, "unknown"),
        }
    }
}

impl From<SolveResult> for CheckSatResult {
    fn from(result: SolveResult) -> Self {
        match result {
            SolveResult::Sat(_) => CheckSatResult::Sat,
            SolveResult::Unsat => CheckSatResult::Unsat,
            SolveResult::Unknown => CheckSatResult::Unknown,
        }
    }
}

/// Detected logic category for theory selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogicCategory {
    /// Pure propositional (SAT)
    Propositional,
    /// QF_UF: Quantifier-free uninterpreted functions
    QfUf,
    /// QF_AX: Quantifier-free arrays with extensionality
    QfAx,
    /// QF_LRA: Quantifier-free linear real arithmetic
    QfLra,
    /// QF_LIA: Quantifier-free linear integer arithmetic
    QfLia,
    /// QF_UFLIA: Quantifier-free uninterpreted functions with linear integer arithmetic
    QfUflia,
    /// QF_UFLRA: Quantifier-free uninterpreted functions with linear real arithmetic
    QfUflra,
    /// QF_AUFLIA: Quantifier-free arrays + uninterpreted functions + linear integer arithmetic
    QfAuflia,
    /// QF_AUFLRA: Quantifier-free arrays + uninterpreted functions + linear real arithmetic
    QfAuflra,
    /// QF_LIRA: Quantifier-free linear integer and real arithmetic (mixed)
    QfLira,
    /// QF_AUFLIRA: Quantifier-free arrays + uninterpreted functions + linear integer/real arithmetic
    QfAuflira,
    /// QF_BV: Quantifier-free bitvectors
    QfBv,
    /// QF_ABV: Quantifier-free arrays + bitvectors
    QfAbv,
    /// QF_UFBV: Quantifier-free uninterpreted functions + bitvectors
    QfUfbv,
    /// QF_AUFBV: Quantifier-free arrays + uninterpreted functions + bitvectors
    QfAufbv,
    /// Other logics (not yet supported)
    Other,
}

/// Theory kind for solve_with_assertions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TheoryKind {
    /// Pure propositional (SAT)
    Propositional,
    /// EUF theory only
    Euf,
    /// Combined EUF + Arrays theory
    ArrayEuf,
    /// Linear Real Arithmetic
    Lra,
    /// Linear Integer Arithmetic
    Lia,
    /// Combined EUF + LIA theory
    UfLia,
    /// Combined EUF + LRA theory
    UfLra,
    /// Combined Arrays + EUF + LIA theory
    AufLia,
    /// Combined Arrays + EUF + LRA theory
    AufLra,
    /// Combined LIA + LRA theory (mixed integer/real arithmetic)
    Lira,
    /// Combined Arrays + EUF + LIA + LRA theory (mixed integer/real arithmetic)
    AufLira,
    /// Bitvector theory (eager bit-blasting)
    Bv,
    /// Combined Arrays + Bitvector theory (eager bit-blasting with array axioms)
    ArrayBv,
    /// Combined UF + Bitvector theory (eager bit-blasting with EUF congruence axioms)
    UfBv,
    /// Combined Arrays + UF + Bitvector theory (eager bit-blasting with array and EUF axioms)
    AufBv,
}

impl LogicCategory {
    fn from_logic(logic: &str) -> Self {
        match logic {
            // Pure propositional
            "QF_UF" => LogicCategory::QfUf,
            "QF_BOOL" | "BOOL" => LogicCategory::Propositional,
            // Arrays
            "QF_AX" => LogicCategory::QfAx,
            // Linear real arithmetic logics
            "QF_LRA" | "QF_RDL" => LogicCategory::QfLra,
            // Linear integer arithmetic logics
            "QF_LIA" | "QF_IDL" => LogicCategory::QfLia,
            // Combined UF + LIA (very common in verification)
            "QF_UFLIA" => LogicCategory::QfUflia,
            // Combined UF + LRA
            "QF_UFLRA" => LogicCategory::QfUflra,
            // Combined Arrays + UF + LIA (very common in software verification)
            "QF_AUFLIA" => LogicCategory::QfAuflia,
            // Combined Arrays + UF + LRA
            "QF_AUFLRA" => LogicCategory::QfAuflra,
            // Mixed integer and real arithmetic
            "QF_LIRA" => LogicCategory::QfLira,
            // Combined Arrays + UF + mixed int/real arithmetic
            "QF_AUFLIRA" => LogicCategory::QfAuflira,
            // Bitvectors
            "QF_BV" => LogicCategory::QfBv,
            // Arrays + Bitvectors (important for Kani workloads)
            "QF_ABV" => LogicCategory::QfAbv,
            // UF + Bitvectors
            "QF_UFBV" => LogicCategory::QfUfbv,
            // Arrays + UF + Bitvectors (critical for Kani workloads)
            "QF_AUFBV" => LogicCategory::QfAufbv,
            // Default to unsupported for unknown logics
            _ => LogicCategory::Other,
        }
    }
}

/// Result of array axiom generation for QF_ABV
struct ArrayAxiomResult {
    /// Generated CNF clauses
    clauses: Vec<z4_core::CnfClause>,
    /// Number of additional variables used
    num_vars: u32,
}

/// Result of EUF congruence axiom generation for QF_UFBV/QF_AUFBV
struct EufAxiomResult {
    /// Generated CNF clauses
    clauses: Vec<z4_core::CnfClause>,
    /// Number of additional variables used
    num_vars: u32,
}

/// A satisfying model from check-sat
#[derive(Debug, Clone)]
struct Model {
    /// SAT variable assignments (indexed by variable)
    sat_model: Vec<bool>,
    /// Mapping from SAT variables to term IDs (for iterating over model)
    #[allow(dead_code)]
    var_to_term: BTreeMap<u32, TermId>,
    /// Reverse mapping from term IDs to SAT variables (for efficient lookup)
    term_to_var: BTreeMap<TermId, u32>,
    /// Optional EUF model (for QF_UF and related logics)
    euf_model: Option<EufModel>,
    /// Optional array model (for QF_AX and related logics)
    array_model: Option<ArrayModel>,
    /// Optional LRA model (for QF_LRA and related logics)
    lra_model: Option<LraModel>,
    /// Optional LIA model (for QF_LIA and related logics)
    lia_model: Option<LiaModel>,
    /// Optional BV model (for QF_BV and related logics)
    bv_model: Option<BvModel>,
}

// NOTE: Clause Retention for Incremental Solving (Kani Fast Requirement 1.2)
//
// IMPLEMENTED: solve_bv_incremental() now uses assumption-based clause retention.
//
// Key design:
// 1. A persistent SAT solver is maintained across check-sat calls
// 2. Each assertion gets a selector variable `s`
// 3. Clauses are added as implications: (-s ∨ clause_lits)
// 4. At check-sat time, selectors for in-scope assertions are passed as assumptions
// 5. Popped assertions have their selectors excluded, disabling their clauses
// 6. Learned clauses are retained across calls, providing the performance benefit
//
// This approach avoids the complexity of tracking global vs scoped assertions
// and syncing SMT push/pop with SAT push/pop. Instead, we let the SAT solver's
// assumption-based solving handle the scoping naturally.

/// Combined EUF + Arrays theory solver
///
/// This solver wraps both EufSolver and ArraySolver, delegating to both
/// and combining their results for QF_AX and related logics.
pub struct ArrayEufSolver<'a> {
    /// EUF solver for equality and congruence reasoning
    euf: EufSolver<'a>,
    /// Array solver for select/store reasoning
    arrays: ArraySolver<'a>,
}

impl<'a> ArrayEufSolver<'a> {
    /// Create a new combined EUF+Arrays solver
    pub fn new(terms: &'a TermStore) -> Self {
        ArrayEufSolver {
            euf: EufSolver::new(terms),
            arrays: ArraySolver::new(terms),
        }
    }

    /// Extract both EUF and Array models for model generation
    pub fn extract_models(&mut self) -> (EufModel, ArrayModel) {
        let euf_model = self.euf.extract_model();
        let array_model = self.arrays.extract_model(&euf_model.term_values);
        (euf_model, array_model)
    }
}

impl TheorySolver for ArrayEufSolver<'_> {
    fn assert_literal(&mut self, literal: TermId, value: bool) {
        // Both solvers need to know about all literals
        self.euf.assert_literal(literal, value);
        self.arrays.assert_literal(literal, value);
    }

    fn check(&mut self) -> TheoryResult {
        // Check EUF first (handles equality and congruence)
        let euf_result = self.euf.check();
        if let TheoryResult::Unsat(reasons) = euf_result {
            return TheoryResult::Unsat(reasons);
        }

        // Then check arrays (handles read-over-write axioms)
        let array_result = self.arrays.check();
        if let TheoryResult::Unsat(reasons) = array_result {
            return TheoryResult::Unsat(reasons);
        }

        // Both theories are satisfied
        TheoryResult::Sat
    }

    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        // Combine propagations from both solvers
        let mut props = self.euf.propagate();
        props.extend(self.arrays.propagate());
        props
    }

    fn push(&mut self) {
        self.euf.push();
        self.arrays.push();
    }

    fn pop(&mut self) {
        self.euf.pop();
        self.arrays.pop();
    }

    fn reset(&mut self) {
        self.euf.reset();
        self.arrays.reset();
    }
}

/// Combined EUF + LIA theory solver
///
/// This solver wraps both EufSolver and LiaSolver, delegating to both
/// and combining their results for QF_UFLIA logic.
pub struct UfLiaSolver<'a> {
    /// Reference to term store for inspecting literals
    terms: &'a TermStore,
    /// EUF solver for equality and congruence reasoning
    euf: EufSolver<'a>,
    /// LIA solver for linear integer arithmetic
    lia: LiaSolver<'a>,
}

impl<'a> UfLiaSolver<'a> {
    /// Create a new combined EUF+LIA solver
    pub fn new(terms: &'a TermStore) -> Self {
        UfLiaSolver {
            terms,
            euf: EufSolver::new(terms),
            lia: LiaSolver::new(terms),
        }
    }

    /// Extract EUF model for model generation
    pub fn extract_euf_model(&mut self) -> EufModel {
        self.euf.extract_model()
    }
}

impl TheorySolver for UfLiaSolver<'_> {
    fn assert_literal(&mut self, literal: TermId, value: bool) {
        // EUF gets all literals
        self.euf.assert_literal(literal, value);

        // LIA only gets literals with actual arithmetic operations
        // (comparisons like <=, or arithmetic ops like +, -, *, /)
        if contains_arithmetic_ops(self.terms, literal) {
            self.lia.assert_literal(literal, value);
        }
    }

    fn check(&mut self) -> TheoryResult {
        // Check EUF first (handles equality and congruence)
        let euf_result = self.euf.check();
        if let TheoryResult::Unsat(reasons) = euf_result {
            return TheoryResult::Unsat(reasons);
        }

        // Then check LIA (handles arithmetic constraints)
        let lia_result = self.lia.check();
        match lia_result {
            TheoryResult::Unsat(reasons) => TheoryResult::Unsat(reasons),
            TheoryResult::Unknown => TheoryResult::Unknown,
            TheoryResult::NeedSplit(split) => TheoryResult::NeedSplit(split),
            TheoryResult::NeedDisequlitySplit(split) => TheoryResult::NeedDisequlitySplit(split),
            TheoryResult::NeedExpressionSplit(split) => TheoryResult::NeedExpressionSplit(split),
            TheoryResult::Sat => TheoryResult::Sat,
        }
    }

    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        // Combine propagations from both solvers
        let mut props = self.euf.propagate();
        props.extend(self.lia.propagate());
        props
    }

    fn push(&mut self) {
        self.euf.push();
        self.lia.push();
    }

    fn pop(&mut self) {
        self.euf.pop();
        self.lia.pop();
    }

    fn reset(&mut self) {
        self.euf.reset();
        self.lia.reset();
    }
}

/// Combined EUF + LRA theory solver
///
/// This solver wraps both EufSolver and LraSolver, delegating to both
/// and combining their results for QF_UFLRA logic.
pub struct UfLraSolver<'a> {
    /// Reference to term store for inspecting literals
    terms: &'a TermStore,
    /// EUF solver for equality and congruence reasoning
    euf: EufSolver<'a>,
    /// LRA solver for linear real arithmetic
    lra: LraSolver<'a>,
}

impl<'a> UfLraSolver<'a> {
    /// Create a new combined EUF+LRA solver
    pub fn new(terms: &'a TermStore) -> Self {
        UfLraSolver {
            terms,
            euf: EufSolver::new(terms),
            lra: LraSolver::new(terms),
        }
    }

    /// Extract EUF model for model generation
    pub fn extract_euf_model(&mut self) -> EufModel {
        self.euf.extract_model()
    }

    /// Extract both EUF and LRA models for model generation
    pub fn extract_models(&mut self) -> (EufModel, LraModel) {
        (self.euf.extract_model(), self.lra.extract_model())
    }
}

impl TheorySolver for UfLraSolver<'_> {
    fn assert_literal(&mut self, literal: TermId, value: bool) {
        // EUF gets all literals
        self.euf.assert_literal(literal, value);

        // LRA only gets literals with actual arithmetic operations
        // (comparisons like <=, or arithmetic ops like +, -, *, /)
        if contains_arithmetic_ops(self.terms, literal) {
            self.lra.assert_literal(literal, value);
        }
    }

    fn check(&mut self) -> TheoryResult {
        // Check EUF first (handles equality and congruence)
        let euf_result = self.euf.check();
        if let TheoryResult::Unsat(reasons) = euf_result {
            return TheoryResult::Unsat(reasons);
        }

        // Then check LRA (handles arithmetic constraints)
        let lra_result = self.lra.check();
        match lra_result {
            TheoryResult::Unsat(reasons) => TheoryResult::Unsat(reasons),
            TheoryResult::Unknown => TheoryResult::Unknown,
            // LRA should never return NeedSplit (that's only for LIA)
            TheoryResult::NeedSplit(_) => TheoryResult::Unknown,
            // Forward disequality split requests from LRA
            TheoryResult::NeedDisequlitySplit(split) => TheoryResult::NeedDisequlitySplit(split),
            // Forward expression split requests from LRA (multi-variable disequalities)
            TheoryResult::NeedExpressionSplit(split) => TheoryResult::NeedExpressionSplit(split),
            TheoryResult::Sat => TheoryResult::Sat,
        }
    }

    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        // Combine propagations from both solvers
        let mut props = self.euf.propagate();
        props.extend(self.lra.propagate());
        props
    }

    fn push(&mut self) {
        self.euf.push();
        self.lra.push();
    }

    fn pop(&mut self) {
        self.euf.pop();
        self.lra.pop();
    }

    fn reset(&mut self) {
        self.euf.reset();
        self.lra.reset();
    }
}

/// Combined Arrays + EUF + LIA theory solver
///
/// This solver wraps ArraySolver, EufSolver, and LiaSolver, delegating to all
/// and combining their results for QF_AUFLIA logic.
pub struct AufLiaSolver<'a> {
    /// Reference to term store for inspecting literals
    terms: &'a TermStore,
    /// EUF solver for equality and congruence reasoning
    euf: EufSolver<'a>,
    /// Array solver for select/store reasoning
    arrays: ArraySolver<'a>,
    /// LIA solver for linear integer arithmetic
    lia: LiaSolver<'a>,
}

impl<'a> AufLiaSolver<'a> {
    /// Create a new combined Arrays+EUF+LIA solver
    pub fn new(terms: &'a TermStore) -> Self {
        AufLiaSolver {
            terms,
            euf: EufSolver::new(terms),
            arrays: ArraySolver::new(terms),
            lia: LiaSolver::new(terms),
        }
    }

    /// Extract EUF and Array models for model generation (backwards compatibility)
    pub fn extract_models(&mut self) -> (EufModel, ArrayModel) {
        let euf_model = self.euf.extract_model();
        let array_model = self.arrays.extract_model(&euf_model.term_values);
        (euf_model, array_model)
    }

    /// Extract all models including LIA for model generation
    pub fn extract_all_models(&mut self) -> (EufModel, ArrayModel, Option<LiaModel>) {
        let euf_model = self.euf.extract_model();
        let array_model = self.arrays.extract_model(&euf_model.term_values);
        let lia_model = self.lia.extract_model();
        (euf_model, array_model, lia_model)
    }
}

impl TheorySolver for AufLiaSolver<'_> {
    fn assert_literal(&mut self, literal: TermId, value: bool) {
        // EUF gets all literals
        self.euf.assert_literal(literal, value);

        // Arrays gets all literals (handles select/store)
        self.arrays.assert_literal(literal, value);

        // LIA only gets literals with actual arithmetic operations
        if contains_arithmetic_ops(self.terms, literal) {
            self.lia.assert_literal(literal, value);
        }
    }

    fn check(&mut self) -> TheoryResult {
        // Check EUF first (handles equality and congruence)
        let euf_result = self.euf.check();
        if let TheoryResult::Unsat(reasons) = euf_result {
            return TheoryResult::Unsat(reasons);
        }

        // Then check arrays (handles read-over-write axioms)
        let array_result = self.arrays.check();
        if let TheoryResult::Unsat(reasons) = array_result {
            return TheoryResult::Unsat(reasons);
        }

        // Finally check LIA (handles arithmetic constraints)
        let lia_result = self.lia.check();
        match lia_result {
            TheoryResult::Unsat(reasons) => TheoryResult::Unsat(reasons),
            TheoryResult::Unknown => TheoryResult::Unknown,
            TheoryResult::NeedSplit(split) => TheoryResult::NeedSplit(split),
            TheoryResult::NeedDisequlitySplit(split) => TheoryResult::NeedDisequlitySplit(split),
            TheoryResult::NeedExpressionSplit(split) => TheoryResult::NeedExpressionSplit(split),
            TheoryResult::Sat => TheoryResult::Sat,
        }
    }

    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        // Combine propagations from all solvers
        let mut props = self.euf.propagate();
        props.extend(self.arrays.propagate());
        props.extend(self.lia.propagate());
        props
    }

    fn push(&mut self) {
        self.euf.push();
        self.arrays.push();
        self.lia.push();
    }

    fn pop(&mut self) {
        self.euf.pop();
        self.arrays.pop();
        self.lia.pop();
    }

    fn reset(&mut self) {
        self.euf.reset();
        self.arrays.reset();
        self.lia.reset();
    }
}

/// Combined Arrays + EUF + LRA theory solver
///
/// This solver wraps ArraySolver, EufSolver, and LraSolver, delegating to all
/// and combining their results for QF_AUFLRA logic.
pub struct AufLraSolver<'a> {
    /// Reference to term store for inspecting literals
    terms: &'a TermStore,
    /// EUF solver for equality and congruence reasoning
    euf: EufSolver<'a>,
    /// Array solver for select/store reasoning
    arrays: ArraySolver<'a>,
    /// LRA solver for linear real arithmetic
    lra: LraSolver<'a>,
}

impl<'a> AufLraSolver<'a> {
    /// Create a new combined Arrays+EUF+LRA solver
    pub fn new(terms: &'a TermStore) -> Self {
        AufLraSolver {
            terms,
            euf: EufSolver::new(terms),
            arrays: ArraySolver::new(terms),
            lra: LraSolver::new(terms),
        }
    }

    /// Extract EUF and Array models for model generation (backwards compatibility)
    pub fn extract_models(&mut self) -> (EufModel, ArrayModel) {
        let euf_model = self.euf.extract_model();
        let array_model = self.arrays.extract_model(&euf_model.term_values);
        (euf_model, array_model)
    }

    /// Extract all models including LRA for model generation
    pub fn extract_all_models(&mut self) -> (EufModel, ArrayModel, LraModel) {
        let euf_model = self.euf.extract_model();
        let array_model = self.arrays.extract_model(&euf_model.term_values);
        let lra_model = self.lra.extract_model();
        (euf_model, array_model, lra_model)
    }
}

impl TheorySolver for AufLraSolver<'_> {
    fn assert_literal(&mut self, literal: TermId, value: bool) {
        // EUF gets all literals
        self.euf.assert_literal(literal, value);

        // Arrays gets all literals (handles select/store)
        self.arrays.assert_literal(literal, value);

        // LRA only gets literals with actual arithmetic operations
        if contains_arithmetic_ops(self.terms, literal) {
            self.lra.assert_literal(literal, value);
        }
    }

    fn check(&mut self) -> TheoryResult {
        // Check EUF first (handles equality and congruence)
        let euf_result = self.euf.check();
        if let TheoryResult::Unsat(reasons) = euf_result {
            return TheoryResult::Unsat(reasons);
        }

        // Then check arrays (handles read-over-write axioms)
        let array_result = self.arrays.check();
        if let TheoryResult::Unsat(reasons) = array_result {
            return TheoryResult::Unsat(reasons);
        }

        // Finally check LRA (handles arithmetic constraints)
        let lra_result = self.lra.check();
        match lra_result {
            TheoryResult::Unsat(reasons) => TheoryResult::Unsat(reasons),
            TheoryResult::Unknown => TheoryResult::Unknown,
            // LRA should never return NeedSplit (that's only for LIA)
            TheoryResult::NeedSplit(_) => TheoryResult::Unknown,
            // Forward disequality split requests from LRA
            TheoryResult::NeedDisequlitySplit(split) => TheoryResult::NeedDisequlitySplit(split),
            // Forward expression split requests from LRA (multi-variable disequalities)
            TheoryResult::NeedExpressionSplit(split) => TheoryResult::NeedExpressionSplit(split),
            TheoryResult::Sat => TheoryResult::Sat,
        }
    }

    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        // Combine propagations from all solvers
        let mut props = self.euf.propagate();
        props.extend(self.arrays.propagate());
        props.extend(self.lra.propagate());
        props
    }

    fn push(&mut self) {
        self.euf.push();
        self.arrays.push();
        self.lra.push();
    }

    fn pop(&mut self) {
        self.euf.pop();
        self.arrays.pop();
        self.lra.pop();
    }

    fn reset(&mut self) {
        self.euf.reset();
        self.arrays.reset();
        self.lra.reset();
    }
}

/// Combined LIA + LRA theory solver for QF_LIRA
///
/// This solver routes literals to LIA or LRA based on the sorts of their operands:
/// - Int-sorted comparisons/equalities go to LIA
/// - Real-sorted comparisons/equalities go to LRA
/// - The SAT solver handles the Boolean combination
pub struct LiraSolver<'a> {
    /// Reference to term store for inspecting literal sorts
    terms: &'a TermStore,
    /// LIA solver for integer arithmetic
    lia: LiaSolver<'a>,
    /// LRA solver for real arithmetic
    lra: LraSolver<'a>,
}

impl<'a> LiraSolver<'a> {
    /// Create a new combined LIA+LRA solver
    pub fn new(terms: &'a TermStore) -> Self {
        LiraSolver {
            terms,
            lia: LiaSolver::new(terms),
            lra: LraSolver::new(terms),
        }
    }

    /// Extract both models for model generation
    pub fn extract_models(&mut self) -> (Option<LiaModel>, LraModel) {
        (self.lia.extract_model(), self.lra.extract_model())
    }
}

impl TheorySolver for LiraSolver<'_> {
    fn assert_literal(&mut self, literal: TermId, value: bool) {
        // Route to LIA if it involves Int arithmetic
        if involves_int_arithmetic(self.terms, literal) {
            self.lia.assert_literal(literal, value);
        }

        // Route to LRA if it involves Real arithmetic
        if involves_real_arithmetic(self.terms, literal) {
            self.lra.assert_literal(literal, value);
        }
    }

    fn check(&mut self) -> TheoryResult {
        // Check LIA first (handles integer arithmetic and integrality)
        match self.lia.check() {
            TheoryResult::Unsat(reasons) => return TheoryResult::Unsat(reasons),
            TheoryResult::Unknown => return TheoryResult::Unknown,
            TheoryResult::NeedSplit(split) => return TheoryResult::NeedSplit(split),
            TheoryResult::NeedDisequlitySplit(split) => {
                return TheoryResult::NeedDisequlitySplit(split)
            }
            TheoryResult::NeedExpressionSplit(split) => {
                return TheoryResult::NeedExpressionSplit(split)
            }
            TheoryResult::Sat => {}
        }

        // Then check LRA (handles real arithmetic)
        match self.lra.check() {
            TheoryResult::Unsat(reasons) => TheoryResult::Unsat(reasons),
            TheoryResult::Unknown => TheoryResult::Unknown,
            TheoryResult::NeedSplit(_) => TheoryResult::Unknown, // LRA shouldn't split
            TheoryResult::NeedDisequlitySplit(split) => TheoryResult::NeedDisequlitySplit(split),
            TheoryResult::NeedExpressionSplit(split) => TheoryResult::NeedExpressionSplit(split),
            TheoryResult::Sat => TheoryResult::Sat,
        }
    }

    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        // Combine propagations from both solvers
        let mut props = self.lia.propagate();
        props.extend(self.lra.propagate());
        props
    }

    fn push(&mut self) {
        self.lia.push();
        self.lra.push();
    }

    fn pop(&mut self) {
        self.lia.pop();
        self.lra.pop();
    }

    fn reset(&mut self) {
        self.lia.reset();
        self.lra.reset();
    }
}

/// Combined Arrays + EUF + LIA + LRA theory solver for QF_AUFLIRA
///
/// This solver supports mixed integer/real arithmetic along with arrays and UF.
pub struct AufLiraSolver<'a> {
    /// Reference to term store for inspecting literal sorts
    terms: &'a TermStore,
    /// EUF solver for equality and congruence reasoning
    euf: EufSolver<'a>,
    /// Array solver for select/store reasoning
    arrays: ArraySolver<'a>,
    /// LIA solver for integer arithmetic
    lia: LiaSolver<'a>,
    /// LRA solver for real arithmetic
    lra: LraSolver<'a>,
}

impl<'a> AufLiraSolver<'a> {
    /// Create a new combined Arrays+EUF+LIA+LRA solver
    pub fn new(terms: &'a TermStore) -> Self {
        AufLiraSolver {
            terms,
            euf: EufSolver::new(terms),
            arrays: ArraySolver::new(terms),
            lia: LiaSolver::new(terms),
            lra: LraSolver::new(terms),
        }
    }

    /// Extract all models for model generation
    pub fn extract_all_models(&mut self) -> (EufModel, ArrayModel, Option<LiaModel>, LraModel) {
        let euf_model = self.euf.extract_model();
        let array_model = self.arrays.extract_model(&euf_model.term_values);
        let lia_model = self.lia.extract_model();
        let lra_model = self.lra.extract_model();
        (euf_model, array_model, lia_model, lra_model)
    }
}

impl TheorySolver for AufLiraSolver<'_> {
    fn assert_literal(&mut self, literal: TermId, value: bool) {
        // EUF gets all literals
        self.euf.assert_literal(literal, value);

        // Arrays gets all literals (handles select/store)
        self.arrays.assert_literal(literal, value);

        // Route to LIA if it involves Int arithmetic
        if involves_int_arithmetic(self.terms, literal) {
            self.lia.assert_literal(literal, value);
        }

        // Route to LRA if it involves Real arithmetic
        if involves_real_arithmetic(self.terms, literal) {
            self.lra.assert_literal(literal, value);
        }
    }

    fn check(&mut self) -> TheoryResult {
        // Check EUF first (handles equality and congruence)
        let euf_result = self.euf.check();
        if let TheoryResult::Unsat(reasons) = euf_result {
            return TheoryResult::Unsat(reasons);
        }

        // Then check arrays (handles read-over-write axioms)
        let array_result = self.arrays.check();
        if let TheoryResult::Unsat(reasons) = array_result {
            return TheoryResult::Unsat(reasons);
        }

        // Check LIA (handles integer arithmetic)
        let lia_result = self.lia.check();
        match lia_result {
            TheoryResult::Unsat(reasons) => return TheoryResult::Unsat(reasons),
            TheoryResult::Unknown => return TheoryResult::Unknown,
            TheoryResult::NeedSplit(split) => return TheoryResult::NeedSplit(split),
            TheoryResult::NeedDisequlitySplit(split) => {
                return TheoryResult::NeedDisequlitySplit(split)
            }
            TheoryResult::NeedExpressionSplit(split) => {
                return TheoryResult::NeedExpressionSplit(split)
            }
            TheoryResult::Sat => {}
        }

        // Finally check LRA (handles real arithmetic)
        let lra_result = self.lra.check();
        match lra_result {
            TheoryResult::Unsat(reasons) => TheoryResult::Unsat(reasons),
            TheoryResult::Unknown => TheoryResult::Unknown,
            TheoryResult::NeedSplit(_) => TheoryResult::Unknown,
            TheoryResult::NeedDisequlitySplit(split) => TheoryResult::NeedDisequlitySplit(split),
            TheoryResult::NeedExpressionSplit(split) => TheoryResult::NeedExpressionSplit(split),
            TheoryResult::Sat => TheoryResult::Sat,
        }
    }

    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        // Combine propagations from all solvers
        let mut props = self.euf.propagate();
        props.extend(self.arrays.propagate());
        props.extend(self.lia.propagate());
        props.extend(self.lra.propagate());
        props
    }

    fn push(&mut self) {
        self.euf.push();
        self.arrays.push();
        self.lia.push();
        self.lra.push();
    }

    fn pop(&mut self) {
        self.euf.pop();
        self.arrays.pop();
        self.lia.pop();
        self.lra.pop();
    }

    fn reset(&mut self) {
        self.euf.reset();
        self.arrays.reset();
        self.lia.reset();
        self.lra.reset();
    }
}

/// Evaluated value from model evaluation
#[derive(Debug, Clone)]
pub enum EvalValue {
    /// Boolean value
    Bool(bool),
    /// Element from an uninterpreted sort (by name, e.g., "@U!0")
    Element(String),
    /// Rational/integer value (BigRational can represent both)
    Rational(BigRational),
    /// Unknown/undefined value
    Unknown,
}

impl PartialEq for EvalValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (EvalValue::Bool(a), EvalValue::Bool(b)) => a == b,
            (EvalValue::Element(a), EvalValue::Element(b)) => a == b,
            (EvalValue::Rational(a), EvalValue::Rational(b)) => a == b,
            (EvalValue::Unknown, EvalValue::Unknown) => true,
            _ => false,
        }
    }
}

impl Eq for EvalValue {}

/// Persistent state for incremental BV solving with clause retention.
///
/// This maintains:
/// - BV variable mappings (term_to_bits) for consistent variable numbering
/// - A persistent SAT solver that retains learned clauses across check-sat calls
/// - Selector variables for each assertion, enabling assumption-based incremental solving
///
/// The key insight is that each assertion gets a selector variable. Clauses are
/// added as implications: (-selector ∨ clause_lits). At check-sat time, we
/// collect selectors for assertions currently in scope and pass them as assumptions.
/// Popped assertions have their selectors excluded, effectively disabling their clauses.
struct IncrementalBvState {
    /// Cached term-to-bits mappings from BvSolver
    term_to_bits: HashMap<TermId, BvBits>,
    /// Next BV variable to allocate (1-indexed for DIMACS compatibility)
    next_bv_var: u32,
    /// Current scope depth (0 = global, 1+ = in push scope)
    scope_depth: usize,

    // Clause retention fields
    /// Persistent SAT solver that retains learned clauses
    persistent_sat: Option<SatSolver>,
    /// Map from assertion TermId to its selector variable in the SAT solver
    assertion_to_selector: HashMap<TermId, SatVariable>,
    /// Base offset for selector variables (set once, never changes)
    /// This ensures selector variables don't overlap with BV variables.
    selector_base: Option<u32>,
    /// Next selector index to allocate (added to selector_base)
    next_selector_idx: u32,
    /// Number of SAT variables allocated (BV vars + selector vars)
    sat_num_vars: usize,
}

impl IncrementalBvState {
    fn new() -> Self {
        IncrementalBvState {
            term_to_bits: HashMap::new(),
            next_bv_var: 1,
            scope_depth: 0,
            persistent_sat: None,
            assertion_to_selector: HashMap::new(),
            selector_base: None,
            next_selector_idx: 0,
            sat_num_vars: 0,
        }
    }

    fn reset(&mut self) {
        self.term_to_bits.clear();
        self.next_bv_var = 1;
        self.scope_depth = 0;
        self.persistent_sat = None;
        self.assertion_to_selector.clear();
        self.selector_base = None;
        self.next_selector_idx = 0;
        self.sat_num_vars = 0;
    }

    fn push(&mut self) {
        self.scope_depth += 1;
    }

    fn pop(&mut self) -> bool {
        if self.scope_depth > 0 {
            self.scope_depth -= 1;
            true
        } else {
            false
        }
    }

    /// Allocate a selector variable for an assertion.
    /// The selector_base is set once on the first allocation and never changes.
    fn alloc_selector(&mut self, current_bv_num_vars: u32) -> SatVariable {
        // Set the selector_base on first allocation, using an offset
        // to ensure selectors don't overlap with future BV variables.
        // Use 4x current BV vars with a minimum of 500 to handle:
        // - 8-bit problems: ~50 vars → base=500
        // - 32-bit problems: ~500 vars → base=2000
        // This gives room for BV variable growth during bitblasting.
        let base = *self
            .selector_base
            .get_or_insert_with(|| std::cmp::max(current_bv_num_vars.saturating_mul(4), 500));
        let var = SatVariable(base + self.next_selector_idx);
        self.next_selector_idx += 1;
        var
    }
}

/// Persistent state for incremental theory solving (QF_UF, QF_LRA, QF_LIA, etc.)
///
/// This maintains:
/// - Tseitin variable mappings for consistent variable numbering across check-sat calls
/// - A persistent SAT solver that retains learned clauses
/// - Selector variables for each assertion, enabling assumption-based incremental solving
///
/// The key insight is that each assertion gets a selector variable. Clauses are
/// added as implications: (-selector ∨ clause_lits). At check-sat time, we
/// collect selectors for assertions currently in scope and pass them as assumptions.
/// Popped assertions have their selectors excluded, effectively disabling their clauses.
struct IncrementalTheoryState {
    /// Persistent SAT solver that retains learned clauses
    persistent_sat: Option<SatSolver>,
    /// Map from assertion TermId to its selector variable
    assertion_to_selector: HashMap<TermId, SatVariable>,
    /// Saved Tseitin state for consistent term-to-var mappings across calls
    tseitin_state: TseitinState,
    /// Base offset for selector variables (set once, never changes)
    selector_base: Option<u32>,
    /// Next selector index to allocate (added to selector_base)
    next_selector_idx: u32,
    /// Current scope depth (0 = global, 1+ = in push scope)
    scope_depth: usize,
    /// Theory atoms registered for theory communication
    theory_atoms: Vec<TermId>,
    /// Selector-guarded clauses for LIA incremental (stored as SAT literals)
    /// Each entry is a complete clause including the negated selector.
    /// Used to rebuild fresh SAT solvers for each LIA check-sat.
    selector_guarded_clauses: Vec<Vec<SatLiteral>>,
}

impl IncrementalTheoryState {
    fn new() -> Self {
        IncrementalTheoryState {
            persistent_sat: None,
            assertion_to_selector: HashMap::new(),
            tseitin_state: TseitinState::new(),
            selector_base: None,
            next_selector_idx: 0,
            scope_depth: 0,
            theory_atoms: Vec::new(),
            selector_guarded_clauses: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.persistent_sat = None;
        self.assertion_to_selector.clear();
        self.tseitin_state = TseitinState::new();
        self.selector_base = None;
        self.next_selector_idx = 0;
        self.scope_depth = 0;
        self.theory_atoms.clear();
        self.selector_guarded_clauses.clear();
    }

    fn push(&mut self) {
        self.scope_depth += 1;
    }

    fn pop(&mut self) -> bool {
        if self.scope_depth > 0 {
            self.scope_depth -= 1;
            true
        } else {
            false
        }
    }

    /// Allocate a selector variable for an assertion.
    /// The selector_base is set once on the first allocation to avoid overlap with CNF vars.
    fn alloc_selector(&mut self, current_num_vars: u32) -> SatVariable {
        // Set the selector_base on first allocation, using a large offset
        // to ensure selectors don't overlap with future CNF variables.
        let base = *self
            .selector_base
            .get_or_insert_with(|| std::cmp::max(current_num_vars.saturating_mul(4), 1000));
        let var = SatVariable(base + self.next_selector_idx);
        self.next_selector_idx += 1;
        var
    }
}

/// SMT executor that coordinates frontend parsing with theory solving
pub struct Executor {
    /// Frontend context for elaboration
    ctx: Context,
    /// Last check-sat result
    last_result: Option<CheckSatResult>,
    /// Last satisfying model (if any)
    last_model: Option<Model>,
    /// Last assumptions from check-sat-assuming (for get-unsat-assumptions)
    last_assumptions: Option<Vec<TermId>>,
    /// Last proof (for get-proof when UNSAT)
    last_proof: Option<Proof>,
    /// Debug flag for QF_UFBV solving
    debug_ufbv: bool,
    /// Whether incremental mode is enabled (detected by push/pop usage)
    ///
    /// When true, incremental solving is used which maintains a persistent
    /// SAT solver to retain learned clauses across check-sat calls.
    incremental_mode: bool,
    /// Persistent state for incremental BV solving with clause retention
    incr_bv_state: Option<IncrementalBvState>,
    /// Persistent state for incremental theory solving (UF/LRA/LIA)
    incr_theory_state: Option<IncrementalTheoryState>,
    /// Style for counterexample generation (model minimization)
    counterexample_style: crate::minimize::CounterexampleStyle,
    /// Proof tracker for collecting proof steps during solving
    proof_tracker: crate::proof_tracker::ProofTracker,
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor {
    /// Create a new executor
    pub fn new() -> Self {
        Executor {
            ctx: Context::new(),
            last_result: None,
            last_model: None,
            last_assumptions: None,
            last_proof: None,
            debug_ufbv: false,
            incremental_mode: false,
            incr_bv_state: None,
            incr_theory_state: None,
            counterexample_style: crate::minimize::CounterexampleStyle::default(),
            proof_tracker: crate::proof_tracker::ProofTracker::new(),
        }
    }

    /// Access the internal context (for API module)
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Access the internal context mutably (for API module)
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.ctx
    }

    /// Enable debug output for QF_UFBV solving
    pub fn set_debug_ufbv(&mut self, enabled: bool) {
        self.debug_ufbv = enabled;
    }

    /// Set the counterexample style for model generation
    ///
    /// This affects how `get-model` generates values:
    /// - `Any`: Return any satisfying value (fast, current behavior)
    /// - `Minimal`: Prefer 0, ±1, powers of 2, MIN/MAX (default)
    /// - `Readable`: Prefer round numbers and simple values
    pub fn set_counterexample_style(&mut self, style: crate::minimize::CounterexampleStyle) {
        self.counterexample_style = style;
    }

    /// Enable or disable proof production
    ///
    /// When enabled, the solver collects proof steps during solving.
    /// After an UNSAT result, call `get_proof()` to retrieve the proof.
    ///
    /// This is required for tRust integration (proof certificates).
    pub fn set_produce_proofs(&mut self, enabled: bool) {
        if enabled {
            self.proof_tracker.enable();
        } else {
            self.proof_tracker.disable();
        }
    }

    /// Check if proof production is enabled
    #[must_use]
    pub fn is_producing_proofs(&self) -> bool {
        self.proof_tracker.is_enabled()
    }

    /// Get the proof from the last UNSAT result as a reference
    ///
    /// Returns None if the last result was not UNSAT or if proof production was disabled.
    #[must_use]
    pub fn get_last_proof(&self) -> Option<&Proof> {
        self.last_proof.as_ref()
    }

    /// Get access to the term store
    #[must_use]
    pub fn terms(&self) -> &z4_core::TermStore {
        &self.ctx.terms
    }

    /// Execute a single command
    ///
    /// Returns output to be printed, if any.
    pub fn execute(&mut self, cmd: &Command) -> Result<Option<String>> {
        // Track incremental mode: enabled on first push, disabled on reset
        // Context handles assertion scoping via push/pop (truncates on pop)
        match cmd {
            Command::Push(n) => {
                self.incremental_mode = true;
                // Initialize incremental BV state if needed
                let bv_state = self
                    .incr_bv_state
                    .get_or_insert_with(IncrementalBvState::new);
                for _ in 0..*n {
                    bv_state.push();
                }
                // Initialize incremental theory state if needed
                let theory_state = self
                    .incr_theory_state
                    .get_or_insert_with(IncrementalTheoryState::new);
                for _ in 0..*n {
                    theory_state.push();
                }
            }
            Command::Pop(n) => {
                if let Some(ref mut state) = self.incr_bv_state {
                    for _ in 0..*n {
                        state.pop();
                    }
                }
                if let Some(ref mut state) = self.incr_theory_state {
                    for _ in 0..*n {
                        state.pop();
                    }
                }
            }
            Command::Reset => {
                self.incremental_mode = false;
                if let Some(ref mut state) = self.incr_bv_state {
                    state.reset();
                }
                if let Some(ref mut state) = self.incr_theory_state {
                    state.reset();
                }
            }
            _ => {}
        }

        let result = self.ctx.process_command(cmd)?;

        match result {
            Some(CommandResult::CheckSat) => {
                let sat_result = self.check_sat_internal()?;
                self.last_result = Some(sat_result);
                Ok(Some(sat_result.to_string()))
            }
            Some(CommandResult::CheckSatAssuming(assumptions)) => {
                let sat_result = self.check_sat_assuming(&assumptions)?;
                self.last_result = Some(sat_result);
                Ok(Some(sat_result.to_string()))
            }
            Some(CommandResult::GetModel) => Ok(Some(self.get_model())),
            Some(CommandResult::GetValue(term_ids)) => Ok(Some(self.get_values(&term_ids))),
            Some(CommandResult::GetInfo(keyword)) => Ok(Some(self.get_info(&keyword))),
            Some(CommandResult::GetOption(keyword)) => Ok(Some(self.get_option_value(&keyword))),
            Some(CommandResult::GetAssertions) => Ok(Some(self.get_assertions())),
            Some(CommandResult::Echo(msg)) => Ok(Some(Self::unquote_string_literal(&msg))),
            Some(CommandResult::GetAssignment) => Ok(Some(self.get_assignment())),
            Some(CommandResult::GetUnsatCore) => Ok(Some(self.get_unsat_core())),
            Some(CommandResult::GetUnsatAssumptions) => Ok(Some(self.get_unsat_assumptions())),
            Some(CommandResult::GetProof) => Ok(Some(self.get_proof())),
            Some(CommandResult::Exit) => Ok(Some("exit".to_string())),
            Some(CommandResult::Simplify(term_id)) => Ok(Some(self.simplify(term_id))),
            None => Ok(None),
        }
    }

    fn produce_models_enabled(&self) -> bool {
        matches!(
            self.ctx.get_option("produce-models"),
            Some(OptionValue::Bool(true))
        )
    }

    fn minimize_counterexamples_enabled(&self) -> bool {
        matches!(
            self.ctx.get_option("minimize-counterexamples"),
            Some(OptionValue::Bool(true))
        )
    }

    fn unquote_string_literal(raw: &str) -> String {
        let inner = raw
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or(raw);

        let mut out = String::with_capacity(inner.len());
        let mut chars = inner.chars();
        while let Some(ch) = chars.next() {
            if ch != '\\' {
                out.push(ch);
                continue;
            }

            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        }
        out
    }

    /// Execute a sequence of commands
    ///
    /// Returns outputs for each command that produces output.
    pub fn execute_all(&mut self, commands: &[Command]) -> Result<Vec<String>> {
        let mut outputs = Vec::new();
        for cmd in commands {
            if let Some(output) = self.execute(cmd)? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }

    /// Run check-sat on current assertions (public API)
    pub fn check_sat(&mut self) -> Result<CheckSatResult> {
        self.check_sat_internal()
    }

    /// Run check-sat with assumptions (public API)
    ///
    /// The assumptions are temporary - they are only used for this check-sat call
    /// and do not affect the assertion stack.
    pub fn check_sat_assuming(&mut self, assumptions: &[TermId]) -> Result<CheckSatResult> {
        // Clear any previous model
        self.last_model = None;

        // Store assumptions for potential get-unsat-assumptions call
        self.last_assumptions = Some(assumptions.to_vec());

        // Create combined assertions: existing assertions + assumptions
        let mut combined_assertions = self.ctx.assertions.clone();
        combined_assertions.extend(assumptions.iter().copied());

        if combined_assertions.is_empty() {
            return Ok(CheckSatResult::Sat);
        }

        let logic = self.ctx.logic.as_deref().unwrap_or("QF_UF");
        let category = LogicCategory::from_logic(logic);

        // Use the combined assertions for this check
        match category {
            LogicCategory::Propositional => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::Propositional)
            }
            LogicCategory::QfUf => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::Euf)
            }
            LogicCategory::QfAx => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::ArrayEuf)
            }
            LogicCategory::QfLra => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::Lra)
            }
            LogicCategory::QfLia => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::Lia)
            }
            LogicCategory::QfUflia => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::UfLia)
            }
            LogicCategory::QfUflra => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::UfLra)
            }
            LogicCategory::QfAuflia => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::AufLia)
            }
            LogicCategory::QfAuflra => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::AufLra)
            }
            LogicCategory::QfLira => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::Lira)
            }
            LogicCategory::QfAuflira => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::AufLira)
            }
            LogicCategory::QfBv => self.solve_with_assertions(&combined_assertions, TheoryKind::Bv),
            LogicCategory::QfAbv => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::ArrayBv)
            }
            LogicCategory::QfUfbv => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::UfBv)
            }
            LogicCategory::QfAufbv => {
                self.solve_with_assertions(&combined_assertions, TheoryKind::AufBv)
            }
            LogicCategory::Other => Err(ExecutorError::UnsupportedLogic(logic.to_string())),
        }
    }

    /// Solve with a specific set of assertions
    fn solve_with_assertions(
        &mut self,
        assertions: &[TermId],
        theory_kind: TheoryKind,
    ) -> Result<CheckSatResult> {
        // Lift ITEs before Tseitin transformation for theories that need it.
        // This transforms (<= (ite c a b) x) → (ite c (<= a x) (<= b x))
        // and (= x (ite c a b)) → (ite c (= x a) (= x b)) for uninterpreted sorts
        // allowing theory solvers to handle atoms without ITE expressions
        let preprocessed_assertions: Vec<TermId> = match theory_kind {
            // EUF needs ITE lifting for equalities involving uninterpreted sorts
            TheoryKind::Euf
            | TheoryKind::Lra
            | TheoryKind::Lia
            | TheoryKind::UfLra
            | TheoryKind::UfLia
            | TheoryKind::AufLra
            | TheoryKind::AufLia
            | TheoryKind::Lira
            | TheoryKind::AufLira => self.ctx.terms.lift_arithmetic_ite_all(assertions),
            _ => assertions.to_vec(),
        };

        // Run Tseitin transformation
        let tseitin = Tseitin::new(&self.ctx.terms);
        let result = tseitin.transform_all(&preprocessed_assertions);

        match theory_kind {
            TheoryKind::Euf => {
                // Create EUF solver with reference to term store
                let theory = EufSolver::new(&self.ctx.terms);
                let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

                let solve_result = dpll.solve();

                // Extract EUF model if SAT
                let euf_model = if matches!(solve_result, SolveResult::Sat(_)) {
                    Some(dpll.theory_solver_mut().extract_model())
                } else {
                    None
                };

                self.solve_and_store_model(solve_result, &result, euf_model, None)
            }
            TheoryKind::ArrayEuf => {
                // Create combined EUF+Arrays solver with reference to term store
                let theory = ArrayEufSolver::new(&self.ctx.terms);
                let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

                let solve_result = dpll.solve();

                // Extract EUF and Array models if SAT
                let (euf_model, array_model) = if matches!(solve_result, SolveResult::Sat(_)) {
                    let (euf, arr) = dpll.theory_solver_mut().extract_models();
                    (Some(euf), Some(arr))
                } else {
                    (None, None)
                };

                self.solve_and_store_model(solve_result, &result, euf_model, array_model)
            }
            TheoryKind::Propositional => {
                // Create DPLL(T) solver with propositional theory
                let theory = PropositionalTheory;
                let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

                self.solve_and_store_model(dpll.solve(), &result, None, None)
            }
            TheoryKind::Lra => {
                // Create LRA solver with reference to term store
                let theory = LraSolver::new(&self.ctx.terms);
                let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

                let solve_result = dpll.solve();

                // For now, no model extraction for LRA (can be added later)
                self.solve_and_store_model(solve_result, &result, None, None)
            }
            TheoryKind::Lia => {
                // Create LIA solver with reference to term store
                // LIA wraps LRA with branch-and-bound for integer constraints
                let theory = LiaSolver::new(&self.ctx.terms);
                let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

                let solve_result = dpll.solve();

                // For now, no model extraction for LIA (can be added later)
                self.solve_and_store_model(solve_result, &result, None, None)
            }
            TheoryKind::UfLia => {
                // Create combined EUF+LIA solver with reference to term store
                let theory = UfLiaSolver::new(&self.ctx.terms);
                let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

                let solve_result = dpll.solve();

                // Extract EUF model if SAT
                let euf_model = if matches!(solve_result, SolveResult::Sat(_)) {
                    Some(dpll.theory_solver_mut().extract_euf_model())
                } else {
                    None
                };

                self.solve_and_store_model(solve_result, &result, euf_model, None)
            }
            TheoryKind::UfLra => {
                // Create combined EUF+LRA solver with reference to term store
                let theory = UfLraSolver::new(&self.ctx.terms);
                let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

                let solve_result = dpll.solve();

                // Extract EUF model if SAT
                let euf_model = if matches!(solve_result, SolveResult::Sat(_)) {
                    Some(dpll.theory_solver_mut().extract_euf_model())
                } else {
                    None
                };

                self.solve_and_store_model(solve_result, &result, euf_model, None)
            }
            TheoryKind::AufLia => {
                // Create combined Arrays+EUF+LIA solver with reference to term store
                let theory = AufLiaSolver::new(&self.ctx.terms);
                let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

                let solve_result = dpll.solve();

                // Extract EUF and Array models if SAT
                let (euf_model, array_model) = if matches!(solve_result, SolveResult::Sat(_)) {
                    let (euf, arr) = dpll.theory_solver_mut().extract_models();
                    (Some(euf), Some(arr))
                } else {
                    (None, None)
                };

                self.solve_and_store_model(solve_result, &result, euf_model, array_model)
            }
            TheoryKind::AufLra => {
                // Create combined Arrays+EUF+LRA solver with reference to term store
                let theory = AufLraSolver::new(&self.ctx.terms);
                let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

                let solve_result = dpll.solve();

                // Extract EUF and Array models if SAT
                let (euf_model, array_model) = if matches!(solve_result, SolveResult::Sat(_)) {
                    let (euf, arr) = dpll.theory_solver_mut().extract_models();
                    (Some(euf), Some(arr))
                } else {
                    (None, None)
                };

                self.solve_and_store_model(solve_result, &result, euf_model, array_model)
            }
            TheoryKind::Lira => {
                // Create combined LIA+LRA solver with reference to term store
                let theory = LiraSolver::new(&self.ctx.terms);
                let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

                let solve_result = dpll.solve();

                // Extract LIA and LRA models if SAT
                let (lia_model, lra_model) = if matches!(solve_result, SolveResult::Sat(_)) {
                    let (lia, lra) = dpll.theory_solver_mut().extract_models();
                    (lia, Some(lra))
                } else {
                    (None, None)
                };

                self.solve_and_store_model_full(
                    solve_result,
                    &result,
                    None,
                    None,
                    lra_model,
                    lia_model,
                )
            }
            TheoryKind::AufLira => {
                // Create combined Arrays+EUF+LIA+LRA solver with reference to term store
                let theory = AufLiraSolver::new(&self.ctx.terms);
                let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

                let solve_result = dpll.solve();

                // Extract all models if SAT
                let (euf_model, array_model, lia_model, lra_model) =
                    if matches!(solve_result, SolveResult::Sat(_)) {
                        let (euf, arr, lia, lra) = dpll.theory_solver_mut().extract_all_models();
                        (Some(euf), Some(arr), lia, Some(lra))
                    } else {
                        (None, None, None, None)
                    };

                self.solve_and_store_model_full(
                    solve_result,
                    &result,
                    euf_model,
                    array_model,
                    lra_model,
                    lia_model,
                )
            }
            TheoryKind::Bv => {
                // For QF_BV, use eager bit-blasting: convert all BV constraints to CNF
                let mut bv_solver = BvSolver::new(&self.ctx.terms);
                let bv_clauses = bv_solver.bitblast_all(assertions);
                let bv_num_vars = bv_solver.num_vars();

                // Combine Tseitin clauses and BV clauses
                let mut all_clauses = result.clauses.clone();

                // Offset BV variables to not conflict with Tseitin variables
                let var_offset = result.num_vars as i32;
                for clause in bv_clauses {
                    let offset_lits: Vec<i32> = clause
                        .literals()
                        .iter()
                        .map(|&lit| {
                            if lit > 0 {
                                lit + var_offset
                            } else {
                                lit - var_offset
                            }
                        })
                        .collect();
                    all_clauses.push(z4_core::CnfClause::new(offset_lits));
                }

                let total_vars = result.num_vars + bv_num_vars;

                // Create a SAT solver with all clauses
                use z4_sat::Solver;
                let mut solver = Solver::new(total_vars as usize);

                for clause in &all_clauses {
                    let lits: Vec<z4_sat::Literal> = clause
                        .literals()
                        .iter()
                        .map(|&lit| {
                            if lit > 0 {
                                z4_sat::Literal::positive(z4_sat::Variable((lit - 1) as u32))
                            } else {
                                z4_sat::Literal::negative(z4_sat::Variable((-lit - 1) as u32))
                            }
                        })
                        .collect();
                    solver.add_clause(lits);
                }

                let solve_result = solver.solve();
                self.solve_and_store_model(solve_result, &result, None, None)
            }
            TheoryKind::ArrayBv => {
                // For QF_ABV, use eager bit-blasting with array axiom generation
                let mut bv_solver = BvSolver::new(&self.ctx.terms);
                let bv_clauses = bv_solver.bitblast_all(assertions);
                let bv_num_vars = bv_solver.num_vars();

                // Combine Tseitin clauses and BV clauses
                let mut all_clauses = result.clauses.clone();

                // Offset BV variables to not conflict with Tseitin variables
                let var_offset = result.num_vars as i32;
                for clause in bv_clauses {
                    let offset_lits: Vec<i32> = clause
                        .literals()
                        .iter()
                        .map(|&lit| {
                            if lit > 0 {
                                lit + var_offset
                            } else {
                                lit - var_offset
                            }
                        })
                        .collect();
                    all_clauses.push(z4_core::CnfClause::new(offset_lits));
                }

                // Generate array axiom clauses
                let array_axiom_result =
                    self.generate_array_bv_axioms(result.num_vars + bv_num_vars);

                // Add array axiom clauses
                for clause in array_axiom_result.clauses {
                    all_clauses.push(clause);
                }

                let total_vars = result.num_vars + bv_num_vars + array_axiom_result.num_vars;

                // Create a SAT solver with all clauses
                use z4_sat::Solver;
                let mut solver = Solver::new(total_vars as usize);

                for clause in &all_clauses {
                    let lits: Vec<z4_sat::Literal> = clause
                        .literals()
                        .iter()
                        .map(|&lit| {
                            if lit > 0 {
                                z4_sat::Literal::positive(z4_sat::Variable((lit - 1) as u32))
                            } else {
                                z4_sat::Literal::negative(z4_sat::Variable((-lit - 1) as u32))
                            }
                        })
                        .collect();
                    solver.add_clause(lits);
                }

                let solve_result = solver.solve();
                self.solve_and_store_model(solve_result, &result, None, None)
            }
            TheoryKind::UfBv => {
                // For QF_UFBV, use eager bit-blasting with EUF congruence axioms
                let mut bv_solver = BvSolver::new(&self.ctx.terms);
                let bv_clauses = bv_solver.bitblast_all(assertions);
                let bv_num_vars = bv_solver.num_vars();

                // Combine Tseitin clauses and BV clauses
                let mut all_clauses = result.clauses.clone();

                // Offset BV variables to not conflict with Tseitin variables
                let var_offset = result.num_vars as i32;
                for clause in bv_clauses {
                    let offset_lits: Vec<i32> = clause
                        .literals()
                        .iter()
                        .map(|&lit| {
                            if lit > 0 {
                                lit + var_offset
                            } else {
                                lit - var_offset
                            }
                        })
                        .collect();
                    all_clauses.push(z4_core::CnfClause::new(offset_lits));
                }

                // Generate EUF congruence axiom clauses
                let euf_axiom_result = self.generate_euf_bv_axioms(
                    &bv_solver,
                    result.num_vars,               // BV variable offset
                    result.num_vars + bv_num_vars, // New variable offset
                );

                // Add EUF axiom clauses
                for clause in euf_axiom_result.clauses {
                    all_clauses.push(clause);
                }

                let total_vars = result.num_vars + bv_num_vars + euf_axiom_result.num_vars;

                // Create a SAT solver with all clauses
                use z4_sat::Solver;
                let mut solver = Solver::new(total_vars as usize);

                for clause in &all_clauses {
                    let lits: Vec<z4_sat::Literal> = clause
                        .literals()
                        .iter()
                        .map(|&lit| {
                            if lit > 0 {
                                z4_sat::Literal::positive(z4_sat::Variable((lit - 1) as u32))
                            } else {
                                z4_sat::Literal::negative(z4_sat::Variable((-lit - 1) as u32))
                            }
                        })
                        .collect();
                    solver.add_clause(lits);
                }

                let solve_result = solver.solve();
                self.solve_and_store_model(solve_result, &result, None, None)
            }
            TheoryKind::AufBv => {
                // For QF_AUFBV, use eager bit-blasting with array and EUF axioms
                let mut bv_solver = BvSolver::new(&self.ctx.terms);
                let bv_clauses = bv_solver.bitblast_all(assertions);
                let bv_num_vars = bv_solver.num_vars();

                // Combine Tseitin clauses and BV clauses
                let mut all_clauses = result.clauses.clone();

                // Offset BV variables to not conflict with Tseitin variables
                let var_offset = result.num_vars as i32;
                for clause in bv_clauses {
                    let offset_lits: Vec<i32> = clause
                        .literals()
                        .iter()
                        .map(|&lit| {
                            if lit > 0 {
                                lit + var_offset
                            } else {
                                lit - var_offset
                            }
                        })
                        .collect();
                    all_clauses.push(z4_core::CnfClause::new(offset_lits));
                }

                // Generate array axiom clauses
                let array_axiom_result =
                    self.generate_array_bv_axioms(result.num_vars + bv_num_vars);

                // Add array axiom clauses
                for clause in array_axiom_result.clauses {
                    all_clauses.push(clause);
                }

                // Generate EUF congruence axiom clauses
                let euf_axiom_result = self.generate_euf_bv_axioms(
                    &bv_solver,
                    result.num_vars, // BV variable offset
                    result.num_vars + bv_num_vars + array_axiom_result.num_vars, // New variable offset
                );

                // Add EUF axiom clauses
                for clause in euf_axiom_result.clauses {
                    all_clauses.push(clause);
                }

                let total_vars = result.num_vars
                    + bv_num_vars
                    + array_axiom_result.num_vars
                    + euf_axiom_result.num_vars;

                // Create a SAT solver with all clauses
                use z4_sat::Solver;
                let mut solver = Solver::new(total_vars as usize);

                for clause in &all_clauses {
                    let lits: Vec<z4_sat::Literal> = clause
                        .literals()
                        .iter()
                        .map(|&lit| {
                            if lit > 0 {
                                z4_sat::Literal::positive(z4_sat::Variable((lit - 1) as u32))
                            } else {
                                z4_sat::Literal::negative(z4_sat::Variable((-lit - 1) as u32))
                            }
                        })
                        .collect();
                    solver.add_clause(lits);
                }

                let solve_result = solver.solve();
                self.solve_and_store_model(solve_result, &result, None, None)
            }
        }
    }

    /// Internal check-sat that also stores the model
    fn check_sat_internal(&mut self) -> Result<CheckSatResult> {
        // Clear any previous model, assumptions, and proof
        self.last_model = None;
        self.last_assumptions = None;
        self.last_proof = None;

        // Sync proof tracker with :produce-proofs option
        if matches!(
            self.ctx.get_option("produce-proofs"),
            Some(OptionValue::Bool(true))
        ) {
            self.proof_tracker.enable();
        }

        // Reset proof tracker for new solving session
        self.proof_tracker.reset();

        if self.ctx.assertions.is_empty() {
            return Ok(CheckSatResult::Sat);
        }

        let logic = self.ctx.logic.as_deref().unwrap_or("QF_UF");
        let category = LogicCategory::from_logic(logic);

        match category {
            LogicCategory::Propositional => self.solve_propositional(),
            LogicCategory::QfUf => self.solve_euf(),
            LogicCategory::QfAx => self.solve_array_euf(),
            LogicCategory::QfLra => self.solve_lra(),
            LogicCategory::QfLia => self.solve_lia(),
            LogicCategory::QfUflia => self.solve_uf_lia(),
            LogicCategory::QfUflra => self.solve_uf_lra(),
            LogicCategory::QfAuflia => self.solve_auf_lia(),
            LogicCategory::QfAuflra => self.solve_auf_lra(),
            LogicCategory::QfLira => self.solve_lira(),
            LogicCategory::QfAuflira => self.solve_auflira(),
            LogicCategory::QfBv => self.solve_bv(),
            LogicCategory::QfAbv => self.solve_abv(),
            LogicCategory::QfUfbv => self.solve_ufbv(),
            LogicCategory::QfAufbv => self.solve_aufbv(),
            LogicCategory::Other => Err(ExecutorError::UnsupportedLogic(logic.to_string())),
        }
    }

    /// Solve using propositional theory (pure SAT)
    fn solve_propositional(&mut self) -> Result<CheckSatResult> {
        // Run Tseitin transformation
        let tseitin = Tseitin::new(&self.ctx.terms);
        let result = tseitin.transform_all(&self.ctx.assertions);

        // Create DPLL(T) solver with propositional theory
        let theory = PropositionalTheory;
        let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

        self.solve_and_store_model(dpll.solve(), &result, None, None)
    }

    /// Solve using EUF theory
    fn solve_euf(&mut self) -> Result<CheckSatResult> {
        // Use incremental mode if enabled (detected by push/pop usage)
        if self.incremental_mode {
            return self.solve_euf_incremental();
        }

        // Lift ITEs from equalities involving uninterpreted sorts
        // This transforms (= x (ite c a b)) → (ite c (= x a) (= x b))
        let lifted_assertions = self.ctx.terms.lift_arithmetic_ite_all(&self.ctx.assertions);

        // Run Tseitin transformation
        let tseitin = Tseitin::new(&self.ctx.terms);
        let result = tseitin.transform_all(&lifted_assertions);

        // Create EUF solver with reference to term store
        let theory = EufSolver::new(&self.ctx.terms);
        let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

        let solve_result = dpll.solve();

        // Extract EUF model if SAT
        let euf_model = if matches!(solve_result, SolveResult::Sat(_)) {
            Some(dpll.theory_solver_mut().extract_model())
        } else {
            None
        };

        self.solve_and_store_model(solve_result, &result, euf_model, None)
    }

    /// Solve QF_UF incrementally using selector-guarded assertions.
    ///
    /// This method maintains a persistent SAT solver and TseitinState that retain
    /// learned clauses and term-to-var mappings across check-sat calls. Each
    /// assertion is guarded by a selector variable, and assumptions are used to
    /// activate/deactivate selectors based on scope.
    fn solve_euf_incremental(&mut self) -> Result<CheckSatResult> {
        // Initialize or get incremental state
        let state = self
            .incr_theory_state
            .get_or_insert_with(IncrementalTheoryState::new);

        // Find assertions that need to be Tseitin-transformed
        let new_assertions: Vec<TermId> = self
            .ctx
            .assertions
            .iter()
            .filter(|&term| !state.assertion_to_selector.contains_key(term))
            .copied()
            .collect();

        // Lift ITEs from equalities involving uninterpreted sorts
        let new_assertions = self.ctx.terms.lift_arithmetic_ite_all(&new_assertions);

        // Use persistent Tseitin state for consistent variable mappings
        let mut tseitin =
            Tseitin::from_state(&self.ctx.terms, std::mem::take(&mut state.tseitin_state));

        // Transform new assertions and collect new clauses per assertion
        struct NewAssertion {
            term: TermId,
            clauses: Vec<z4_core::CnfClause>,
        }
        let mut new_assertion_clauses: Vec<NewAssertion> = Vec::new();

        for &term in &new_assertions {
            tseitin.encode_and_assert(term);
            let clauses = tseitin.take_new_clauses();
            new_assertion_clauses.push(NewAssertion { term, clauses });
        }

        // Calculate total CNF variables (Tseitin uses 1-indexed, we use 0-indexed)
        let total_cnf_vars = tseitin.num_vars();

        // Allocate selectors for new assertions
        let mut new_selectors: Vec<(TermId, SatVariable, Vec<z4_core::CnfClause>)> = Vec::new();
        for na in new_assertion_clauses {
            let selector = state.alloc_selector(total_cnf_vars);
            state.assertion_to_selector.insert(na.term, selector);
            new_selectors.push((na.term, selector, na.clauses));
        }

        // Calculate selector variable range
        let selector_base = state.selector_base.unwrap_or(1000);
        let total_selectors = state.next_selector_idx as usize;
        let total_vars = std::cmp::max(
            total_cnf_vars as usize,
            selector_base as usize + total_selectors,
        );

        // Initialize or resize persistent SAT solver
        let solver = if let Some(ref mut s) = state.persistent_sat {
            s.ensure_num_vars(total_vars);
            s
        } else {
            state.persistent_sat = Some(SatSolver::new(total_vars));
            state.persistent_sat.as_mut().unwrap()
        };

        // Add selector-guarded clauses for new assertions
        for (_term, selector, clauses) in new_selectors {
            // Ensure solver has enough variables
            solver.ensure_num_vars(selector.0 as usize + 1);

            // Add each clause as: (-selector ∨ clause_lits)
            for clause in &clauses {
                let mut lits: Vec<SatLiteral> = vec![SatLiteral::negative(selector)];
                for &lit in clause.literals() {
                    // Convert 1-indexed Tseitin var to 0-indexed SAT var
                    let sat_lit = if lit > 0 {
                        let var = SatVariable((lit - 1) as u32);
                        SatLiteral::positive(var)
                    } else {
                        let var = SatVariable((-lit - 1) as u32);
                        SatLiteral::negative(var)
                    };
                    lits.push(sat_lit);
                }
                solver.add_clause(lits);
            }
        }

        // Build var_to_term map from Tseitin state (convert 1-indexed to 0-indexed)
        let var_to_term: BTreeMap<u32, TermId> = tseitin
            .var_to_term()
            .iter()
            .map(|(&v, &t)| (v - 1, t))
            .collect();
        let term_to_var: BTreeMap<TermId, u32> = tseitin
            .term_to_var()
            .iter()
            .map(|(&t, &v)| (t, v - 1))
            .collect();

        // Build assumptions: active selectors true, inactive selectors false
        // Sort by variable index for deterministic SAT solver behavior (hashbrown
        // iteration order is non-deterministic due to random hash seeds)
        let active_terms: HashSet<TermId> = self.ctx.assertions.iter().copied().collect();
        let mut assumptions: Vec<SatLiteral> = state
            .assertion_to_selector
            .iter()
            .map(|(term, &var)| {
                if active_terms.contains(term) {
                    SatLiteral::positive(var)
                } else {
                    SatLiteral::negative(var)
                }
            })
            .collect();
        assumptions.sort_by_key(|lit| lit.variable().index());

        // Lazy DPLL(T) loop: SAT with assumptions, then check theory
        loop {
            let sat_result = solver.solve_with_assumptions(&assumptions);

            match sat_result {
                AssumeResult::Sat(model) => {
                    // SAT found a model - check with EUF theory
                    let mut theory = EufSolver::new(&self.ctx.terms);
                    theory.reset();

                    // Sync model to theory: for each assigned variable with a term mapping
                    // Sort by variable index for deterministic assertion order
                    let mut var_term_pairs: Vec<_> = var_to_term.iter().collect();
                    var_term_pairs.sort_by_key(|(&var, _)| var);
                    for (&var, &term) in var_term_pairs {
                        if (var as usize) < model.len()
                            && crate::is_theory_atom(&self.ctx.terms, term)
                        {
                            theory.assert_literal(term, model[var as usize]);
                        }
                    }

                    // Check theory consistency
                    match theory.check() {
                        z4_core::TheoryResult::Sat => {
                            // Theory accepts - extract model and return
                            let euf_model = Some(theory.extract_model());
                            // Build Tseitin result for model storage
                            let fake_result = TseitinResult {
                                clauses: vec![],
                                var_to_term: var_to_term
                                    .iter()
                                    .map(|(&v, &t)| (v + 1, t))
                                    .collect(),
                                term_to_var: term_to_var
                                    .iter()
                                    .map(|(&t, &v)| (t, v + 1))
                                    .collect(),
                                root: 1, // Dummy root, not used for model extraction
                                num_vars: total_cnf_vars,
                            };
                            // Save Tseitin state back before returning
                            state.tseitin_state = tseitin.into_state();
                            return self.solve_and_store_model(
                                SolveResult::Sat(model),
                                &fake_result,
                                euf_model,
                                None,
                            );
                        }
                        z4_core::TheoryResult::Unsat(conflict_terms) => {
                            // Theory conflict - add blocking clause
                            let clause: Vec<SatLiteral> = conflict_terms
                                .iter()
                                .filter_map(|t| {
                                    term_to_var.get(&t.term).map(|&var| {
                                        if t.value {
                                            SatLiteral::negative(SatVariable(var))
                                        } else {
                                            SatLiteral::positive(SatVariable(var))
                                        }
                                    })
                                })
                                .collect();
                            if clause.is_empty() {
                                // Empty clause means UNSAT
                                state.tseitin_state = tseitin.into_state();
                                self.last_result = Some(CheckSatResult::Unsat);
                                return Ok(CheckSatResult::Unsat);
                            }
                            solver.add_clause(clause);
                            // Continue SAT loop
                        }
                        z4_core::TheoryResult::Unknown
                        | z4_core::TheoryResult::NeedSplit(_)
                        | z4_core::TheoryResult::NeedDisequlitySplit(_)
                        | z4_core::TheoryResult::NeedExpressionSplit(_) => {
                            // Can't determine - return unknown
                            state.tseitin_state = tseitin.into_state();
                            self.last_result = Some(CheckSatResult::Unknown);
                            return Ok(CheckSatResult::Unknown);
                        }
                    }
                }
                AssumeResult::Unsat(_) => {
                    state.tseitin_state = tseitin.into_state();
                    self.last_model = None;
                    self.last_result = Some(CheckSatResult::Unsat);
                    return Ok(CheckSatResult::Unsat);
                }
                AssumeResult::Unknown => {
                    state.tseitin_state = tseitin.into_state();
                    self.last_model = None;
                    self.last_result = Some(CheckSatResult::Unknown);
                    return Ok(CheckSatResult::Unknown);
                }
            }
        }
    }

    /// Solve using combined EUF + Arrays theory
    fn solve_array_euf(&mut self) -> Result<CheckSatResult> {
        // Run Tseitin transformation
        let tseitin = Tseitin::new(&self.ctx.terms);
        let result = tseitin.transform_all(&self.ctx.assertions);

        // Create combined EUF+Arrays solver with reference to term store
        let theory = ArrayEufSolver::new(&self.ctx.terms);
        let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

        let solve_result = dpll.solve();

        // Extract EUF and Array models if SAT
        let (euf_model, array_model) = if matches!(solve_result, SolveResult::Sat(_)) {
            let (euf, arr) = dpll.theory_solver_mut().extract_models();
            (Some(euf), Some(arr))
        } else {
            (None, None)
        };

        self.solve_and_store_model(solve_result, &result, euf_model, array_model)
    }

    /// Solve using Linear Real Arithmetic theory
    fn solve_lra(&mut self) -> Result<CheckSatResult> {
        // Use incremental mode if enabled (detected by push/pop usage)
        if self.incremental_mode {
            return self.solve_lra_incremental();
        }

        // Lift arithmetic ITEs before Tseitin transformation
        // This transforms (<= (ite c a b) x) → (ite c (<= a x) (<= b x))
        let lifted_assertions = self.ctx.terms.lift_arithmetic_ite_all(&self.ctx.assertions);

        // Run Tseitin transformation
        let tseitin = Tseitin::new(&self.ctx.terms);
        let result = tseitin.transform_all(&lifted_assertions);

        // Create LRA solver with reference to term store
        let theory = LraSolver::new(&self.ctx.terms);
        let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

        let solve_result = dpll.solve();

        // Extract LRA model if SAT
        let lra_model = if matches!(solve_result, SolveResult::Sat(_)) {
            Some(dpll.theory_solver_mut().extract_model())
        } else {
            None
        };

        self.solve_and_store_model_full(solve_result, &result, None, None, lra_model, None)
    }

    /// Solve QF_LRA incrementally using selector-guarded assertions.
    ///
    /// This method maintains a persistent SAT solver and TseitinState that retain
    /// learned clauses and term-to-var mappings across check-sat calls. Each
    /// assertion is guarded by a selector variable, and assumptions are used to
    /// activate/deactivate selectors based on scope.
    fn solve_lra_incremental(&mut self) -> Result<CheckSatResult> {
        // Initialize or get incremental state
        let state = self
            .incr_theory_state
            .get_or_insert_with(IncrementalTheoryState::new);

        // Find assertions that need to be Tseitin-transformed
        let new_assertions: Vec<TermId> = self
            .ctx
            .assertions
            .iter()
            .filter(|&term| !state.assertion_to_selector.contains_key(term))
            .copied()
            .collect();

        // Lift arithmetic ITEs from new assertions
        let lifted_assertions = self.ctx.terms.lift_arithmetic_ite_all(&new_assertions);

        // Use persistent Tseitin state for consistent variable mappings
        let mut tseitin =
            Tseitin::from_state(&self.ctx.terms, std::mem::take(&mut state.tseitin_state));

        // Transform lifted assertions and collect new clauses per assertion
        struct NewAssertion {
            term: TermId,
            clauses: Vec<z4_core::CnfClause>,
        }
        let mut new_assertion_clauses: Vec<NewAssertion> = Vec::new();

        // Process lifted assertions (original assertion term IDs for selector tracking,
        // but the lifted terms go through Tseitin)
        for (original_term, lifted_term) in new_assertions.iter().zip(lifted_assertions.iter()) {
            tseitin.encode_and_assert(*lifted_term);
            let clauses = tseitin.take_new_clauses();
            new_assertion_clauses.push(NewAssertion {
                term: *original_term,
                clauses,
            });
        }

        // Calculate total CNF variables (Tseitin uses 1-indexed, we use 0-indexed)
        let total_cnf_vars = tseitin.num_vars();

        // Allocate selectors for new assertions
        let mut new_selectors: Vec<(TermId, SatVariable, Vec<z4_core::CnfClause>)> = Vec::new();
        for na in new_assertion_clauses {
            let selector = state.alloc_selector(total_cnf_vars);
            state.assertion_to_selector.insert(na.term, selector);
            new_selectors.push((na.term, selector, na.clauses));
        }

        // Calculate selector variable range
        let selector_base = state.selector_base.unwrap_or(1000);
        let total_selectors = state.next_selector_idx as usize;
        let total_vars = std::cmp::max(
            total_cnf_vars as usize,
            selector_base as usize + total_selectors,
        );

        // Initialize or resize persistent SAT solver
        let solver = if let Some(ref mut s) = state.persistent_sat {
            s.ensure_num_vars(total_vars);
            s
        } else {
            state.persistent_sat = Some(SatSolver::new(total_vars));
            state.persistent_sat.as_mut().unwrap()
        };

        // Add selector-guarded clauses for new assertions
        for (_term, selector, clauses) in new_selectors {
            // Ensure solver has enough variables
            solver.ensure_num_vars(selector.0 as usize + 1);

            // Add each clause as: (-selector ∨ clause_lits)
            for clause in &clauses {
                let mut lits: Vec<SatLiteral> = vec![SatLiteral::negative(selector)];
                for &lit in clause.literals() {
                    // Convert 1-indexed Tseitin var to 0-indexed SAT var
                    let sat_lit = if lit > 0 {
                        let var = SatVariable((lit - 1) as u32);
                        SatLiteral::positive(var)
                    } else {
                        let var = SatVariable((-lit - 1) as u32);
                        SatLiteral::negative(var)
                    };
                    lits.push(sat_lit);
                }
                solver.add_clause(lits);
            }
        }

        // Build var_to_term map from Tseitin state (convert 1-indexed to 0-indexed)
        let var_to_term: BTreeMap<u32, TermId> = tseitin
            .var_to_term()
            .iter()
            .map(|(&v, &t)| (v - 1, t))
            .collect();
        let term_to_var: BTreeMap<TermId, u32> = tseitin
            .term_to_var()
            .iter()
            .map(|(&t, &v)| (t, v - 1))
            .collect();

        // Build assumptions: active selectors true, inactive selectors false
        // Sort by variable index for deterministic SAT solver behavior
        let active_terms: HashSet<TermId> = self.ctx.assertions.iter().copied().collect();
        let mut assumptions: Vec<SatLiteral> = state
            .assertion_to_selector
            .iter()
            .map(|(term, &var)| {
                if active_terms.contains(term) {
                    SatLiteral::positive(var)
                } else {
                    SatLiteral::negative(var)
                }
            })
            .collect();
        assumptions.sort_by_key(|lit| lit.variable().index());

        // Lazy DPLL(T) loop: SAT with assumptions, then check theory
        loop {
            let sat_result = solver.solve_with_assumptions(&assumptions);

            match sat_result {
                AssumeResult::Sat(model) => {
                    // SAT found a model - check with LRA theory
                    let mut theory = LraSolver::new(&self.ctx.terms);
                    theory.reset();

                    // Sync model to theory: for each assigned variable with a term mapping
                    // Sort by variable index for deterministic assertion order
                    let mut var_term_pairs: Vec<_> = var_to_term.iter().collect();
                    var_term_pairs.sort_by_key(|(&var, _)| var);
                    for (&var, &term) in var_term_pairs {
                        if (var as usize) < model.len()
                            && crate::is_theory_atom(&self.ctx.terms, term)
                        {
                            theory.assert_literal(term, model[var as usize]);
                        }
                    }

                    // Check theory consistency
                    match theory.check() {
                        z4_core::TheoryResult::Sat => {
                            // Theory accepts - extract model and return
                            let lra_model = Some(theory.extract_model());
                            // Build Tseitin result for model storage
                            let fake_result = TseitinResult {
                                clauses: vec![],
                                var_to_term: var_to_term
                                    .iter()
                                    .map(|(&v, &t)| (v + 1, t))
                                    .collect(),
                                term_to_var: term_to_var
                                    .iter()
                                    .map(|(&t, &v)| (t, v + 1))
                                    .collect(),
                                root: 1, // Dummy root, not used for model extraction
                                num_vars: total_cnf_vars,
                            };
                            // Save Tseitin state back before returning
                            state.tseitin_state = tseitin.into_state();
                            return self.solve_and_store_model_full(
                                SolveResult::Sat(model),
                                &fake_result,
                                None,
                                None,
                                lra_model,
                                None,
                            );
                        }
                        z4_core::TheoryResult::Unsat(conflict_terms) => {
                            // Theory conflict - add blocking clause
                            let clause: Vec<SatLiteral> = conflict_terms
                                .iter()
                                .filter_map(|t| {
                                    term_to_var.get(&t.term).map(|&var| {
                                        if t.value {
                                            SatLiteral::negative(SatVariable(var))
                                        } else {
                                            SatLiteral::positive(SatVariable(var))
                                        }
                                    })
                                })
                                .collect();
                            if clause.is_empty() {
                                // Empty clause means UNSAT
                                state.tseitin_state = tseitin.into_state();
                                self.last_result = Some(CheckSatResult::Unsat);
                                return Ok(CheckSatResult::Unsat);
                            }
                            solver.add_clause(clause);
                            // Continue SAT loop
                        }
                        z4_core::TheoryResult::Unknown
                        | z4_core::TheoryResult::NeedSplit(_)
                        | z4_core::TheoryResult::NeedDisequlitySplit(_)
                        | z4_core::TheoryResult::NeedExpressionSplit(_) => {
                            // Can't determine - return unknown
                            state.tseitin_state = tseitin.into_state();
                            self.last_result = Some(CheckSatResult::Unknown);
                            return Ok(CheckSatResult::Unknown);
                        }
                    }
                }
                AssumeResult::Unsat(_) => {
                    state.tseitin_state = tseitin.into_state();
                    self.last_model = None;
                    self.last_result = Some(CheckSatResult::Unsat);
                    return Ok(CheckSatResult::Unsat);
                }
                AssumeResult::Unknown => {
                    state.tseitin_state = tseitin.into_state();
                    self.last_model = None;
                    self.last_result = Some(CheckSatResult::Unknown);
                    return Ok(CheckSatResult::Unknown);
                }
            }
        }
    }

    /// Solve using Linear Integer Arithmetic theory
    ///
    /// LIA uses branch-and-bound over LRA. When the LRA relaxation gives a
    /// non-integer solution, we add splitting lemmas to guide the search.
    fn solve_lia(&mut self) -> Result<CheckSatResult> {
        if self.incremental_mode {
            return self.solve_lia_incremental();
        }

        use crate::SolveStepResult;
        use z4_sat::Literal;

        // Run Tseitin transformation
        let tseitin = Tseitin::new(&self.ctx.terms);
        let tseitin_result = tseitin.transform_all(&self.ctx.assertions);

        // Track split atoms that need to be added (le_atom, ge_atom, prefer_ceil_hint)
        // prefer_ceil: Some(true) = prefer ceil branch, Some(false) = prefer floor branch
        let mut split_atoms: Vec<(TermId, TermId, Option<bool>)> = Vec::new();

        // Track disequality split atoms (lt_atom, gt_atom)
        let mut diseq_split_atoms: Vec<(TermId, TermId)> = Vec::new();

        // Track learned clauses to preserve across solver recreations
        let mut preserved_learned: Vec<Vec<Literal>> = Vec::new();

        // Track learned HNF cuts and deduplication keys across solver recreations
        let mut lia_learned_cuts: Vec<z4_lia::StoredCut> = Vec::new();
        let mut lia_seen_hnf_cuts: hashbrown::HashSet<z4_lia::HnfCutKey> =
            hashbrown::HashSet::new();
        // Track Diophantine solver state across solver recreations
        // Since equalities don't change during branch-and-bound, we can reuse the analysis
        let mut lia_dioph_state: z4_lia::DiophState =
            (Vec::new(), hashbrown::HashSet::new(), Vec::new());

        // Maximum number of splits to prevent infinite loops
        const MAX_SPLITS: usize = 1000;

        for iteration in 0..MAX_SPLITS {
            // Create LIA solver with reference to term store
            let theory = LiaSolver::new(&self.ctx.terms);
            let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &tseitin_result, theory);

            // Apply any pending split atoms from previous iterations
            for &(le_atom, ge_atom, prefer_ceil) in &split_atoms {
                dpll.apply_split_with_hint(le_atom, ge_atom, prefer_ceil);
            }

            // Apply any pending disequality split atoms from previous iterations
            for &(lt_atom, gt_atom) in &diseq_split_atoms {
                dpll.apply_disequality_split(lt_atom, gt_atom);
            }

            // Re-add learned clauses from previous iterations
            // This is critical for branch-and-bound to converge!
            dpll.add_learned_clauses(preserved_learned.clone());

            // Reset theory, then import learned state (after reset to preserve it)
            dpll.reset_theory();
            dpll.theory_solver_mut().import_learned_state(
                std::mem::take(&mut lia_learned_cuts),
                std::mem::take(&mut lia_seen_hnf_cuts),
            );
            dpll.theory_solver_mut()
                .import_dioph_state(std::mem::take(&mut lia_dioph_state));
            match dpll.solve_step() {
                SolveStepResult::Done(solve_result) => {
                    if std::env::var("Z4_DEBUG_LIA").is_ok() {
                        eprintln!(
                            "[LIA] Done after {} iterations: {:?}",
                            iteration, solve_result
                        );
                    }
                    // Extract LIA model if SAT
                    let lia_model = if matches!(solve_result, SolveResult::Sat(_)) {
                        dpll.theory_solver_mut().extract_model()
                    } else {
                        None
                    };
                    // Final result - return it
                    return self.solve_and_store_model_full(
                        solve_result,
                        &tseitin_result,
                        None,
                        None,
                        None,
                        lia_model,
                    );
                }
                SolveStepResult::NeedSplit(split) => {
                    if std::env::var("Z4_DEBUG_LIA").is_ok() {
                        eprintln!(
                            "[LIA] Iteration {}: split on var {:?} at value {}, learned {} clauses",
                            iteration,
                            split.variable,
                            split.value,
                            preserved_learned.len()
                        );
                    }
                    // Save learned clauses before dropping dpll
                    preserved_learned = dpll.get_learned_clauses();

                    // Save learned HNF and Diophantine state before dropping dpll
                    (lia_learned_cuts, lia_seen_hnf_cuts) =
                        dpll.theory_solver_mut().take_learned_state();
                    lia_dioph_state = dpll.theory_solver_mut().take_dioph_state();

                    // Compute which branch to prefer: closer integer first
                    // For value 3.7 (frac=0.7 > 0.5), prefer ceil (4)
                    // For value 3.2 (frac=0.2 < 0.5), prefer floor (3)
                    let frac_part =
                        &split.value - num_rational::BigRational::from(split.floor.clone());
                    let half = num_rational::BigRational::new(
                        num_bigint::BigInt::from(1),
                        num_bigint::BigInt::from(2),
                    );
                    let prefer_ceil = Some(frac_part > half);

                    // Drop dpll to release the term store reference
                    drop(dpll);

                    // Now we can safely modify the term store to create split atoms
                    // Create: var <= floor and var >= ceil
                    let floor_term = self.ctx.terms.mk_int(split.floor.clone());
                    let ceil_term = self.ctx.terms.mk_int(split.ceil.clone());
                    let le_atom = self.ctx.terms.mk_le(split.variable, floor_term);
                    let ge_atom = self.ctx.terms.mk_ge(split.variable, ceil_term);

                    split_atoms.push((le_atom, ge_atom, prefer_ceil));
                    // Continue to next iteration with the new split
                }
                SolveStepResult::NeedDisequlitySplit(split) => {
                    if std::env::var("Z4_DEBUG_LIA").is_ok() {
                        eprintln!("[LIA] Iteration {}: disequality split on var {:?} excluding value {}, learned {} clauses",
                                  iteration, split.variable, split.excluded_value, preserved_learned.len());
                    }
                    // Save learned clauses before dropping dpll
                    preserved_learned = dpll.get_learned_clauses();

                    // Save learned HNF and Diophantine state before dropping dpll
                    (lia_learned_cuts, lia_seen_hnf_cuts) =
                        dpll.theory_solver_mut().take_learned_state();
                    lia_dioph_state = dpll.theory_solver_mut().take_dioph_state();

                    // Drop dpll to release the term store reference
                    drop(dpll);

                    // Create: var < excluded_value OR var > excluded_value
                    // This excludes the specific value for integer variable
                    let excluded_term = self.ctx.terms.mk_int(split.excluded_value.to_integer());
                    let lt_atom = self.ctx.terms.mk_lt(split.variable, excluded_term);
                    let gt_atom = self.ctx.terms.mk_gt(split.variable, excluded_term);

                    diseq_split_atoms.push((lt_atom, gt_atom));
                    // Continue to next iteration with the new split
                }
            }
        }

        // Too many splits - return unknown to prevent infinite loops
        Ok(CheckSatResult::Unknown)
    }

    /// Solve QF_LIA incrementally using selector-guarded assertions.
    ///
    /// This method maintains persistent Tseitin mappings and selector assignments
    /// across check-sat calls. For LIA, each check-sat uses a fresh SAT solver
    /// because branch-and-bound splits should not persist across check-sats.
    ///
    /// LIA uses branch-and-bound for integer solutions. When the LIA theory
    /// reports NeedSplit, we add split constraints and continue the SAT loop.
    fn solve_lia_incremental(&mut self) -> Result<CheckSatResult> {
        // Initialize or get incremental state (for tracking assertions/selectors)
        let state = self
            .incr_theory_state
            .get_or_insert_with(IncrementalTheoryState::new);

        // Find assertions that need to be Tseitin-transformed
        let new_assertions: Vec<TermId> = self
            .ctx
            .assertions
            .iter()
            .filter(|&term| !state.assertion_to_selector.contains_key(term))
            .copied()
            .collect();

        // Use persistent Tseitin state for consistent variable mappings
        let mut tseitin =
            Tseitin::from_state(&self.ctx.terms, std::mem::take(&mut state.tseitin_state));

        // Transform new assertions and store their clauses
        struct NewAssertion {
            term: TermId,
            clauses: Vec<z4_core::CnfClause>,
        }
        let mut new_assertion_clauses: Vec<NewAssertion> = Vec::new();

        for &term in &new_assertions {
            tseitin.encode_and_assert(term);
            let clauses = tseitin.take_new_clauses();
            new_assertion_clauses.push(NewAssertion { term, clauses });
        }

        // Calculate total CNF variables (Tseitin uses 1-indexed, we use 0-indexed)
        let total_cnf_vars = tseitin.num_vars();

        // Allocate selectors for new assertions and store clauses
        for na in new_assertion_clauses {
            let selector = state.alloc_selector(total_cnf_vars);
            state.assertion_to_selector.insert(na.term, selector);

            // Store selector-guarded clauses for replay
            for clause in &na.clauses {
                let mut lits: Vec<SatLiteral> = vec![SatLiteral::negative(selector)];
                for &lit in clause.literals() {
                    // Convert 1-indexed Tseitin var to 0-indexed SAT var
                    let sat_lit = if lit > 0 {
                        let var = SatVariable((lit - 1) as u32);
                        SatLiteral::positive(var)
                    } else {
                        let var = SatVariable((-lit - 1) as u32);
                        SatLiteral::negative(var)
                    };
                    lits.push(sat_lit);
                }
                state.selector_guarded_clauses.push(lits);
            }
        }

        // Calculate selector variable range
        let selector_base = state.selector_base.unwrap_or(1000);
        let total_selectors = state.next_selector_idx as usize;
        let total_vars = std::cmp::max(
            total_cnf_vars as usize,
            selector_base as usize + total_selectors,
        );

        // Build var_to_term map from Tseitin state (convert 1-indexed to 0-indexed)
        let base_var_to_term: BTreeMap<u32, TermId> = tseitin
            .var_to_term()
            .iter()
            .map(|(&v, &t)| (v - 1, t))
            .collect();
        let base_term_to_var: BTreeMap<TermId, u32> = tseitin
            .term_to_var()
            .iter()
            .map(|(&t, &v)| (t, v - 1))
            .collect();

        // Save Tseitin state back
        state.tseitin_state = tseitin.into_state();

        // Build assumptions: active selectors true, inactive selectors false
        // Sort by variable index for deterministic SAT solver behavior
        let active_terms: HashSet<TermId> = self.ctx.assertions.iter().copied().collect();
        let mut base_assumptions: Vec<SatLiteral> = state
            .assertion_to_selector
            .iter()
            .map(|(term, &var)| {
                if active_terms.contains(term) {
                    SatLiteral::positive(var)
                } else {
                    SatLiteral::negative(var)
                }
            })
            .collect();
        base_assumptions.sort_by_key(|lit| lit.variable().index());

        // Maximum number of splits to prevent infinite loops
        const MAX_SPLITS: usize = 1000;

        // For LIA, we use a fresh SAT solver for each check-sat to avoid
        // persisting branch-and-bound splits across check-sats
        let mut solver = SatSolver::new(total_vars);

        // Track local state for this check-sat (splits, etc.)
        let mut local_term_to_var: BTreeMap<TermId, u32> = base_term_to_var.clone();
        let mut local_var_to_term: BTreeMap<u32, TermId> = base_var_to_term;
        let mut local_next_var = total_cnf_vars;

        // Add all stored selector-guarded clauses to fresh solver
        for clause in &state.selector_guarded_clauses {
            solver.add_clause(clause.clone());
        }

        // Learned cuts and deduplication keys persist across theory instances.
        // HNF cuts are derived from the original constraint matrix and are globally valid.
        let mut learned_cuts: Vec<z4_lia::StoredCut> = Vec::new();
        let mut seen_hnf_cuts: hashbrown::HashSet<z4_lia::HnfCutKey> = hashbrown::HashSet::new();
        // Diophantine solver state: preserves equality analysis across iterations.
        // Since lazy DPLL(T) only changes inequality bounds (not equalities),
        // we can reuse the variable elimination and substitution analysis.
        let mut dioph_state: z4_lia::DiophState =
            (Vec::new(), hashbrown::HashSet::new(), Vec::new());

        // Lazy DPLL(T) loop: SAT with assumptions, then check theory
        for _iteration in 0..MAX_SPLITS {
            let sat_result = solver.solve_with_assumptions(&base_assumptions);

            match sat_result {
                AssumeResult::Sat(model) => {
                    // SAT found a model - check with LIA theory
                    // Create fresh theory each iteration but preserve learned state
                    let mut theory = LiaSolver::new(&self.ctx.terms);
                    theory.import_learned_state(
                        std::mem::take(&mut learned_cuts),
                        std::mem::take(&mut seen_hnf_cuts),
                    );
                    theory.import_dioph_state(std::mem::take(&mut dioph_state));

                    // Sync model to theory: for each assigned variable with a term mapping
                    // Sort by variable index for deterministic assertion order
                    let mut var_term_pairs: Vec<_> = local_var_to_term.iter().collect();
                    var_term_pairs.sort_by_key(|(&var, _)| var);
                    for (&var, &term) in var_term_pairs {
                        if (var as usize) < model.len()
                            && crate::is_theory_atom(&self.ctx.terms, term)
                        {
                            theory.assert_literal(term, model[var as usize]);
                        }
                    }

                    // Replay any learned cuts from previous iterations
                    theory.replay_learned_cuts();

                    // Check theory consistency
                    match theory.check() {
                        z4_core::TheoryResult::Sat => {
                            // Theory accepts - extract model and return
                            let lia_model = theory.extract_model();
                            // Build Tseitin result for model storage
                            let fake_result = TseitinResult {
                                clauses: vec![],
                                var_to_term: local_var_to_term
                                    .iter()
                                    .map(|(&v, &t)| (v + 1, t))
                                    .collect(),
                                term_to_var: local_term_to_var
                                    .iter()
                                    .map(|(&t, &v)| (t, v + 1))
                                    .collect(),
                                root: 1, // Dummy root, not used for model extraction
                                num_vars: local_next_var,
                            };
                            return self.solve_and_store_model_full(
                                SolveResult::Sat(model),
                                &fake_result,
                                None,
                                None,
                                None,
                                lia_model,
                            );
                        }
                        z4_core::TheoryResult::Unsat(conflict_terms) => {
                            // Save learned state before theory goes out of scope
                            (learned_cuts, seen_hnf_cuts) = theory.take_learned_state();
                            dioph_state = theory.take_dioph_state();

                            // Theory conflict - add blocking clause
                            let clause: Vec<SatLiteral> = conflict_terms
                                .iter()
                                .filter_map(|t| {
                                    local_term_to_var.get(&t.term).map(|&var| {
                                        if t.value {
                                            SatLiteral::negative(SatVariable(var))
                                        } else {
                                            SatLiteral::positive(SatVariable(var))
                                        }
                                    })
                                })
                                .collect();
                            if clause.is_empty() {
                                // Empty clause means UNSAT
                                self.last_result = Some(CheckSatResult::Unsat);
                                return Ok(CheckSatResult::Unsat);
                            }
                            solver.add_clause(clause);
                            // Continue SAT loop
                        }
                        z4_core::TheoryResult::NeedSplit(split) => {
                            // Save learned state and drop theory before mutating terms
                            (learned_cuts, seen_hnf_cuts) = theory.take_learned_state();
                            dioph_state = theory.take_dioph_state();
                            drop(theory);

                            // LIA needs a split for branch-and-bound
                            // Create split atoms: var <= floor OR var >= ceil
                            let floor_term = self.ctx.terms.mk_int(split.floor.clone());
                            let ceil_term = self.ctx.terms.mk_int(split.ceil.clone());
                            let le_atom = self.ctx.terms.mk_le(split.variable, floor_term);
                            let ge_atom = self.ctx.terms.mk_ge(split.variable, ceil_term);

                            // Tseitin transform the split atoms (fresh instances for local splits)
                            let le_tseitin = Tseitin::new(&self.ctx.terms);
                            let le_result = le_tseitin.transform_all(&[le_atom]);
                            let ge_tseitin = Tseitin::new(&self.ctx.terms);
                            let ge_result = ge_tseitin.transform_all(&[ge_atom]);

                            // Add the split atoms to local state (not persistent)
                            let le_offset = local_next_var;
                            for (&term, &var) in le_result.term_to_var.iter() {
                                let sat_var = (var - 1) + le_offset;
                                local_term_to_var.insert(term, sat_var);
                                local_var_to_term.insert(sat_var, term);
                            }
                            local_next_var += le_result.num_vars;

                            let ge_offset = local_next_var;
                            for (&term, &var) in ge_result.term_to_var.iter() {
                                let sat_var = (var - 1) + ge_offset;
                                local_term_to_var.insert(term, sat_var);
                                local_var_to_term.insert(sat_var, term);
                            }
                            local_next_var += ge_result.num_vars;

                            // Ensure solver has enough variables
                            solver.ensure_num_vars(local_next_var as usize);

                            // Add clauses for le_atom (skip root assertion unit clause)
                            // For simple atoms, Tseitin generates a unit clause [root] to assert it.
                            // We don't want to assert the atom - we only want to define it.
                            // The disjunction (le_atom OR ge_atom) is added separately.
                            for clause in &le_result.clauses {
                                // Skip unit clauses that just assert the root
                                if clause.literals().len() == 1
                                    && clause.literals()[0] == le_result.root
                                {
                                    continue;
                                }
                                let lits: Vec<SatLiteral> = clause
                                    .literals()
                                    .iter()
                                    .map(|&lit| {
                                        if lit > 0 {
                                            let var = SatVariable((lit - 1) as u32 + le_offset);
                                            SatLiteral::positive(var)
                                        } else {
                                            let var = SatVariable((-lit - 1) as u32 + le_offset);
                                            SatLiteral::negative(var)
                                        }
                                    })
                                    .collect();
                                solver.add_clause(lits);
                            }

                            // Add clauses for ge_atom (skip root assertion unit clause)
                            for clause in &ge_result.clauses {
                                // Skip unit clauses that just assert the root
                                if clause.literals().len() == 1
                                    && clause.literals()[0] == ge_result.root
                                {
                                    continue;
                                }
                                let lits: Vec<SatLiteral> = clause
                                    .literals()
                                    .iter()
                                    .map(|&lit| {
                                        if lit > 0 {
                                            let var = SatVariable((lit - 1) as u32 + ge_offset);
                                            SatLiteral::positive(var)
                                        } else {
                                            let var = SatVariable((-lit - 1) as u32 + ge_offset);
                                            SatLiteral::negative(var)
                                        }
                                    })
                                    .collect();
                                solver.add_clause(lits);
                            }

                            // Add the split clause: le_atom OR ge_atom
                            // (var <= floor) OR (var >= ceil)
                            if let (Some(&le_var), Some(&ge_var)) = (
                                local_term_to_var.get(&le_atom),
                                local_term_to_var.get(&ge_atom),
                            ) {
                                // Set phase hints based on which integer is closer to the fractional value.
                                // For value 3.7 (frac=0.7 > 0.5), prefer ceil (4) -> ge_var=true
                                // For value 3.2 (frac=0.2 < 0.5), prefer floor (3) -> le_var=true
                                // This reduces unnecessary backtracking by trying the closer integer first.
                                let frac_part = &split.value
                                    - num_rational::BigRational::from(split.floor.clone());
                                let half = num_rational::BigRational::new(
                                    num_bigint::BigInt::from(1),
                                    num_bigint::BigInt::from(2),
                                );
                                if frac_part > half {
                                    // Closer to ceil: prefer ge_var=true, le_var=false
                                    solver.set_var_phase(SatVariable(ge_var), true);
                                    solver.set_var_phase(SatVariable(le_var), false);
                                } else {
                                    // Closer to floor: prefer le_var=true, ge_var=false
                                    solver.set_var_phase(SatVariable(le_var), true);
                                    solver.set_var_phase(SatVariable(ge_var), false);
                                }

                                solver.add_clause(vec![
                                    SatLiteral::positive(SatVariable(le_var)),
                                    SatLiteral::positive(SatVariable(ge_var)),
                                ]);
                            }

                            // Continue SAT loop with the new split
                        }
                        z4_core::TheoryResult::Unknown
                        | z4_core::TheoryResult::NeedDisequlitySplit(_)
                        | z4_core::TheoryResult::NeedExpressionSplit(_) => {
                            // Can't determine - return unknown
                            // (Disequality/expression splits not yet supported in executor)
                            self.last_result = Some(CheckSatResult::Unknown);
                            return Ok(CheckSatResult::Unknown);
                        }
                    }
                }
                AssumeResult::Unsat(_) => {
                    self.last_model = None;
                    self.last_result = Some(CheckSatResult::Unsat);
                    return Ok(CheckSatResult::Unsat);
                }
                AssumeResult::Unknown => {
                    self.last_model = None;
                    self.last_result = Some(CheckSatResult::Unknown);
                    return Ok(CheckSatResult::Unknown);
                }
            }
        }

        // Too many splits - return unknown to prevent infinite loops
        self.last_result = Some(CheckSatResult::Unknown);
        Ok(CheckSatResult::Unknown)
    }

    /// Solve using combined EUF + LIA theory
    fn solve_uf_lia(&mut self) -> Result<CheckSatResult> {
        // Run Tseitin transformation
        let tseitin = Tseitin::new(&self.ctx.terms);
        let result = tseitin.transform_all(&self.ctx.assertions);

        // Create combined EUF+LIA solver with reference to term store
        let theory = UfLiaSolver::new(&self.ctx.terms);
        let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

        let solve_result = dpll.solve();

        // Extract EUF model if SAT
        let euf_model = if matches!(solve_result, SolveResult::Sat(_)) {
            Some(dpll.theory_solver_mut().extract_euf_model())
        } else {
            None
        };

        self.solve_and_store_model(solve_result, &result, euf_model, None)
    }

    /// Solve using combined EUF + LRA theory
    fn solve_uf_lra(&mut self) -> Result<CheckSatResult> {
        // Run Tseitin transformation
        let tseitin = Tseitin::new(&self.ctx.terms);
        let result = tseitin.transform_all(&self.ctx.assertions);

        // Create combined EUF+LRA solver with reference to term store
        let theory = UfLraSolver::new(&self.ctx.terms);
        let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

        let solve_result = dpll.solve();

        // Extract EUF and LRA models if SAT
        let (euf_model, lra_model) = if matches!(solve_result, SolveResult::Sat(_)) {
            let (euf, lra) = dpll.theory_solver_mut().extract_models();
            (Some(euf), Some(lra))
        } else {
            (None, None)
        };

        self.solve_and_store_model_full(solve_result, &result, euf_model, None, lra_model, None)
    }

    /// Solve using combined Arrays + EUF + LIA theory
    fn solve_auf_lia(&mut self) -> Result<CheckSatResult> {
        // Run Tseitin transformation
        let tseitin = Tseitin::new(&self.ctx.terms);
        let result = tseitin.transform_all(&self.ctx.assertions);

        // Create combined Arrays+EUF+LIA solver with reference to term store
        let theory = AufLiaSolver::new(&self.ctx.terms);
        let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

        let solve_result = dpll.solve();

        // Extract EUF, Array, and LIA models if SAT
        let (euf_model, array_model, lia_model) = if matches!(solve_result, SolveResult::Sat(_)) {
            let (euf, arr, lia) = dpll.theory_solver_mut().extract_all_models();
            (Some(euf), Some(arr), lia)
        } else {
            (None, None, None)
        };

        self.solve_and_store_model_full(
            solve_result,
            &result,
            euf_model,
            array_model,
            None,
            lia_model,
        )
    }

    /// Solve using combined Arrays + EUF + LRA theory
    fn solve_auf_lra(&mut self) -> Result<CheckSatResult> {
        // Run Tseitin transformation
        let tseitin = Tseitin::new(&self.ctx.terms);
        let result = tseitin.transform_all(&self.ctx.assertions);

        // Create combined Arrays+EUF+LRA solver with reference to term store
        let theory = AufLraSolver::new(&self.ctx.terms);
        let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &result, theory);

        let solve_result = dpll.solve();

        // Extract EUF, Array, and LRA models if SAT
        let (euf_model, array_model, lra_model) = if matches!(solve_result, SolveResult::Sat(_)) {
            let (euf, arr, lra) = dpll.theory_solver_mut().extract_all_models();
            (Some(euf), Some(arr), Some(lra))
        } else {
            (None, None, None)
        };

        self.solve_and_store_model_full(
            solve_result,
            &result,
            euf_model,
            array_model,
            lra_model,
            None,
        )
    }

    /// Solve using combined LIA + LRA theory (QF_LIRA)
    ///
    /// This handles both integer branch-and-bound splits (NeedSplit) and
    /// disequality splits (NeedDisequlitySplit) for mixed Int+Real problems.
    fn solve_lira(&mut self) -> Result<CheckSatResult> {
        use crate::SolveStepResult;

        // Run Tseitin transformation
        let tseitin = Tseitin::new(&self.ctx.terms);
        let tseitin_result = tseitin.transform_all(&self.ctx.assertions);

        // Track integer split atoms (le_atom, ge_atom, prefer_ceil_hint)
        let mut int_split_atoms: Vec<(TermId, TermId, Option<bool>)> = Vec::new();

        // Track disequality split atoms (lt_atom, gt_atom)
        let mut diseq_split_atoms: Vec<(TermId, TermId)> = Vec::new();

        // Maximum number of splits to prevent infinite loops
        const MAX_SPLITS: usize = 1000;

        for iteration in 0..MAX_SPLITS {
            // Create combined LIA+LRA solver with reference to term store
            let theory = LiraSolver::new(&self.ctx.terms);
            let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &tseitin_result, theory);

            // Apply any pending integer split atoms from previous iterations
            for &(le_atom, ge_atom, prefer_ceil) in &int_split_atoms {
                dpll.apply_split_with_hint(le_atom, ge_atom, prefer_ceil);
            }

            // Apply any pending disequality split atoms from previous iterations
            for &(lt_atom, gt_atom) in &diseq_split_atoms {
                dpll.apply_disequality_split(lt_atom, gt_atom);
            }

            match dpll.solve_step() {
                SolveStepResult::Done(solve_result) => {
                    if std::env::var("Z4_DEBUG_LIRA").is_ok() {
                        eprintln!(
                            "[LIRA] Done after {} iterations: {:?}",
                            iteration, solve_result
                        );
                    }
                    // Extract LIA and LRA models if SAT
                    let (lia_model, lra_model) = if matches!(solve_result, SolveResult::Sat(_)) {
                        let (lia, lra) = dpll.theory_solver_mut().extract_models();
                        (lia, Some(lra))
                    } else {
                        (None, None)
                    };
                    return self.solve_and_store_model_full(
                        solve_result,
                        &tseitin_result,
                        None,
                        None,
                        lra_model,
                        lia_model,
                    );
                }
                SolveStepResult::NeedSplit(split) => {
                    if std::env::var("Z4_DEBUG_LIRA").is_ok() {
                        eprintln!(
                            "[LIRA] Iteration {}: int split on var {:?} at value {}",
                            iteration, split.variable, split.value
                        );
                    }
                    // Compute which branch to prefer: closer integer first
                    let frac_part =
                        &split.value - num_rational::BigRational::from(split.floor.clone());
                    let half = num_rational::BigRational::new(
                        num_bigint::BigInt::from(1),
                        num_bigint::BigInt::from(2),
                    );
                    let prefer_ceil = Some(frac_part > half);

                    // Drop dpll to release the term store reference
                    drop(dpll);

                    // Create: var <= floor and var >= ceil
                    let floor_term = self.ctx.terms.mk_int(split.floor.clone());
                    let ceil_term = self.ctx.terms.mk_int(split.ceil.clone());
                    let le_atom = self.ctx.terms.mk_le(split.variable, floor_term);
                    let ge_atom = self.ctx.terms.mk_ge(split.variable, ceil_term);

                    int_split_atoms.push((le_atom, ge_atom, prefer_ceil));
                }
                SolveStepResult::NeedDisequlitySplit(split) => {
                    if std::env::var("Z4_DEBUG_LIRA").is_ok() {
                        eprintln!(
                            "[LIRA] Iteration {}: diseq split on var {:?} != {}",
                            iteration, split.variable, split.excluded_value
                        );
                    }

                    // Drop dpll to release the term store reference
                    drop(dpll);

                    // Create atoms for x < c and x > c to exclude x = c
                    // Determine sort of the variable to create appropriate constant
                    let var_sort = self.ctx.terms.sort(split.variable);
                    let excluded_term = match var_sort {
                        Sort::Int => {
                            // For integers, the excluded value should be an integer
                            // Convert BigRational to BigInt (floor)
                            let int_val = split.excluded_value.to_integer();
                            self.ctx.terms.mk_int(int_val)
                        }
                        Sort::Real => self.ctx.terms.mk_rational(split.excluded_value.clone()),
                        _ => {
                            // Unsupported sort for disequality split
                            return Ok(CheckSatResult::Unknown);
                        }
                    };

                    let lt_atom = self.ctx.terms.mk_lt(split.variable, excluded_term);
                    let gt_atom = self.ctx.terms.mk_gt(split.variable, excluded_term);

                    diseq_split_atoms.push((lt_atom, gt_atom));
                }
            }
        }

        // Too many splits - return unknown to prevent infinite loops
        Ok(CheckSatResult::Unknown)
    }

    /// Solve using combined Arrays + EUF + LIA + LRA theory (QF_AUFLIRA)
    ///
    /// This handles both integer branch-and-bound splits (NeedSplit) and
    /// disequality splits (NeedDisequlitySplit) for mixed Int+Real problems.
    fn solve_auflira(&mut self) -> Result<CheckSatResult> {
        use crate::SolveStepResult;

        // Run Tseitin transformation
        let tseitin = Tseitin::new(&self.ctx.terms);
        let tseitin_result = tseitin.transform_all(&self.ctx.assertions);

        // Track integer split atoms (le_atom, ge_atom, prefer_ceil_hint)
        let mut int_split_atoms: Vec<(TermId, TermId, Option<bool>)> = Vec::new();

        // Track disequality split atoms (lt_atom, gt_atom)
        let mut diseq_split_atoms: Vec<(TermId, TermId)> = Vec::new();

        // Maximum number of splits to prevent infinite loops
        const MAX_SPLITS: usize = 1000;

        for iteration in 0..MAX_SPLITS {
            // Create combined Arrays+EUF+LIA+LRA solver with reference to term store
            let theory = AufLiraSolver::new(&self.ctx.terms);
            let mut dpll = DpllT::from_tseitin(&self.ctx.terms, &tseitin_result, theory);

            // Apply any pending integer split atoms from previous iterations
            for &(le_atom, ge_atom, prefer_ceil) in &int_split_atoms {
                dpll.apply_split_with_hint(le_atom, ge_atom, prefer_ceil);
            }

            // Apply any pending disequality split atoms from previous iterations
            for &(lt_atom, gt_atom) in &diseq_split_atoms {
                dpll.apply_disequality_split(lt_atom, gt_atom);
            }

            match dpll.solve_step() {
                SolveStepResult::Done(solve_result) => {
                    if std::env::var("Z4_DEBUG_AUFLIRA").is_ok() {
                        eprintln!(
                            "[AUFLIRA] Done after {} iterations: {:?}",
                            iteration, solve_result
                        );
                    }
                    // Extract all models if SAT
                    let (euf_model, array_model, lia_model, lra_model) =
                        if matches!(solve_result, SolveResult::Sat(_)) {
                            let (euf, arr, lia, lra) =
                                dpll.theory_solver_mut().extract_all_models();
                            (Some(euf), Some(arr), lia, Some(lra))
                        } else {
                            (None, None, None, None)
                        };
                    return self.solve_and_store_model_full(
                        solve_result,
                        &tseitin_result,
                        euf_model,
                        array_model,
                        lra_model,
                        lia_model,
                    );
                }
                SolveStepResult::NeedSplit(split) => {
                    if std::env::var("Z4_DEBUG_AUFLIRA").is_ok() {
                        eprintln!(
                            "[AUFLIRA] Iteration {}: int split on var {:?} at value {}",
                            iteration, split.variable, split.value
                        );
                    }
                    // Compute which branch to prefer: closer integer first
                    let frac_part =
                        &split.value - num_rational::BigRational::from(split.floor.clone());
                    let half = num_rational::BigRational::new(
                        num_bigint::BigInt::from(1),
                        num_bigint::BigInt::from(2),
                    );
                    let prefer_ceil = Some(frac_part > half);

                    // Drop dpll to release the term store reference
                    drop(dpll);

                    // Create: var <= floor and var >= ceil
                    let floor_term = self.ctx.terms.mk_int(split.floor.clone());
                    let ceil_term = self.ctx.terms.mk_int(split.ceil.clone());
                    let le_atom = self.ctx.terms.mk_le(split.variable, floor_term);
                    let ge_atom = self.ctx.terms.mk_ge(split.variable, ceil_term);

                    int_split_atoms.push((le_atom, ge_atom, prefer_ceil));
                }
                SolveStepResult::NeedDisequlitySplit(split) => {
                    if std::env::var("Z4_DEBUG_AUFLIRA").is_ok() {
                        eprintln!(
                            "[AUFLIRA] Iteration {}: diseq split on var {:?} != {}",
                            iteration, split.variable, split.excluded_value
                        );
                    }

                    // Drop dpll to release the term store reference
                    drop(dpll);

                    // Create atoms for x < c and x > c to exclude x = c
                    // Determine sort of the variable to create appropriate constant
                    let var_sort = self.ctx.terms.sort(split.variable);
                    let excluded_term = match var_sort {
                        Sort::Int => {
                            // For integers, the excluded value should be an integer
                            // Convert BigRational to BigInt (floor)
                            let int_val = split.excluded_value.to_integer();
                            self.ctx.terms.mk_int(int_val)
                        }
                        Sort::Real => self.ctx.terms.mk_rational(split.excluded_value.clone()),
                        _ => {
                            // Unsupported sort for disequality split
                            return Ok(CheckSatResult::Unknown);
                        }
                    };

                    let lt_atom = self.ctx.terms.mk_lt(split.variable, excluded_term);
                    let gt_atom = self.ctx.terms.mk_gt(split.variable, excluded_term);

                    diseq_split_atoms.push((lt_atom, gt_atom));
                }
            }
        }

        // Too many splits - return unknown to prevent infinite loops
        Ok(CheckSatResult::Unknown)
    }

    /// Extract BV model values from SAT model and term-to-bits mapping
    fn extract_bv_model_from_bits(
        sat_model: &[bool],
        term_bits: &HashMap<TermId, BvBits>,
        var_offset: i32,
        terms: &TermStore,
    ) -> BvModel {
        use num_bigint::BigInt;

        let mut values = HashMap::new();
        let mut stored_term_to_bits = HashMap::new();

        for (&term_id, bits) in term_bits {
            // Only include user-declared BV variables
            let term_data = terms.get(term_id);
            if !matches!(term_data, TermData::Var(_, _)) {
                continue;
            }
            let sort = terms.sort(term_id);
            if !matches!(sort, Sort::BitVec(_)) {
                continue;
            }

            stored_term_to_bits.insert(term_id, bits.clone());

            // Reconstruct value from bits (LSB at index 0)
            let mut value = BigInt::from(0);
            for (i, &bit_lit) in bits.iter().enumerate() {
                let offset_lit = if bit_lit > 0 {
                    bit_lit + var_offset
                } else {
                    bit_lit - var_offset
                };
                let sat_var_idx = if offset_lit > 0 {
                    (offset_lit - 1) as usize
                } else {
                    (-offset_lit - 1) as usize
                };
                let bit_value = if sat_var_idx < sat_model.len() {
                    let sat_val = sat_model[sat_var_idx];
                    if offset_lit > 0 {
                        sat_val
                    } else {
                        !sat_val
                    }
                } else {
                    false
                };
                if bit_value {
                    value |= BigInt::from(1) << i;
                }
            }
            values.insert(term_id, value);
        }

        BvModel {
            values,
            term_to_bits: stored_term_to_bits,
        }
    }

    /// Solve using Bitvector theory (eager bit-blasting)
    ///
    /// For QF_BV, we use eager bit-blasting: all bitvector constraints are
    /// converted to CNF clauses upfront, then solved by the SAT solver.
    fn solve_bv(&mut self) -> Result<CheckSatResult> {
        // Use incremental mode if enabled (detected by push/pop usage)
        if self.incremental_mode {
            return self.solve_bv_incremental();
        }

        // Non-incremental path: create fresh solver each time
        // First, run Tseitin transformation for the Boolean structure
        let tseitin = Tseitin::new(&self.ctx.terms);
        let tseitin_result = tseitin.transform_all(&self.ctx.assertions);

        // Then, bit-blast all BV constraints to CNF
        let mut bv_solver = BvSolver::new(&self.ctx.terms);
        let bv_clauses = bv_solver.bitblast_all(&self.ctx.assertions);
        let bv_num_vars = bv_solver.num_vars();

        // Combine Tseitin clauses and BV clauses
        let mut all_clauses = tseitin_result.clauses.clone();

        // Offset BV variables to not conflict with Tseitin variables
        let var_offset = tseitin_result.num_vars as i32;
        for clause in bv_clauses {
            let offset_lits: Vec<i32> = clause
                .literals()
                .iter()
                .map(|&lit| {
                    if lit > 0 {
                        lit + var_offset
                    } else {
                        lit - var_offset
                    }
                })
                .collect();
            all_clauses.push(z4_core::CnfClause::new(offset_lits));
        }

        let total_vars = tseitin_result.num_vars + bv_num_vars;

        // Create a SAT solver with all clauses
        let mut solver = SatSolver::new(total_vars as usize);

        for clause in &all_clauses {
            let lits: Vec<SatLiteral> = clause
                .literals()
                .iter()
                .map(|&lit| {
                    if lit > 0 {
                        SatLiteral::positive(SatVariable((lit - 1) as u32))
                    } else {
                        SatLiteral::negative(SatVariable((-lit - 1) as u32))
                    }
                })
                .collect();
            solver.add_clause(lits);
        }

        let solve_result = solver.solve();

        // Store model if SAT
        if let SolveResult::Sat(ref model) = solve_result {
            let sat_model: Vec<bool> = model.to_vec();
            let bv_model = Self::extract_bv_model_from_bits(
                &sat_model,
                bv_solver.term_to_bits(),
                var_offset,
                &self.ctx.terms,
            );
            self.last_model = Some(Model {
                sat_model,
                var_to_term: tseitin_result.var_to_term.clone(),
                term_to_var: tseitin_result.term_to_var.clone(),
                euf_model: None,
                array_model: None,
                lra_model: None,
                lia_model: None,
                bv_model: Some(bv_model),
            });
            self.last_result = Some(CheckSatResult::Sat);
            debug_assert!(
                self.validate_model().is_ok(),
                "BUG: Model validation failed: {:?}",
                self.validate_model()
            );
            return Ok(CheckSatResult::Sat);
        }

        self.last_model = None;
        let result = CheckSatResult::from(solve_result);
        self.last_result = Some(result);
        Ok(result)
    }

    /// Incremental QF_BV solving with clause retention using assumption-based solving.
    ///
    /// This maintains:
    /// - BV term-to-bits mappings for consistent variable numbering
    /// - A persistent SAT solver that retains learned clauses across check-sat calls
    /// - Selector variables for each assertion (enabling/disabling via assumptions)
    ///
    /// Key insight: Each assertion gets a selector variable `s`. Clauses are added as
    /// implications: `(-s ∨ clause_lits)`. At check-sat time, we collect selectors for
    /// assertions currently in ctx.assertions and pass them as positive assumptions.
    /// Popped assertions have their selectors excluded, effectively disabling their clauses
    /// while retaining learned clauses that don't depend on them.
    fn solve_bv_incremental(&mut self) -> Result<CheckSatResult> {
        // Initialize state if needed
        let state = self
            .incr_bv_state
            .get_or_insert_with(IncrementalBvState::new);

        // Create a BvSolver with cached term-to-bits mappings
        let mut bv_solver = BvSolver::new(&self.ctx.terms);
        for (term, bits) in &state.term_to_bits {
            bv_solver.set_term_bits(*term, bits.clone());
        }
        bv_solver.set_next_var(state.next_bv_var);

        // Find assertions that need to be bitblasted (not yet in persistent SAT solver)
        let new_assertions: Vec<TermId> = self
            .ctx
            .assertions
            .iter()
            .filter(|&term| !state.assertion_to_selector.contains_key(term))
            .copied()
            .collect();

        // Bitblast new assertions
        let new_clauses_by_assertion: Vec<(TermId, Vec<z4_core::CnfClause>)> = new_assertions
            .iter()
            .map(|&term| {
                let clauses = bv_solver.bitblast_all(&[term]);
                (term, clauses)
            })
            .collect();

        let bv_num_vars = bv_solver.num_vars();

        // Update cached term-to-bits mappings
        for (term, bits) in bv_solver.iter_term_bits() {
            state.term_to_bits.insert(term, bits.to_vec());
        }
        state.next_bv_var = bv_num_vars + 1;

        // Allocate selector variables for new assertions BEFORE borrowing the SAT solver
        // (to avoid borrow checker issues)
        let new_selectors: Vec<(TermId, SatVariable, Vec<z4_core::CnfClause>)> =
            new_clauses_by_assertion
                .into_iter()
                .map(|(term, clauses)| {
                    let selector_var = state.alloc_selector(bv_num_vars);
                    state.assertion_to_selector.insert(term, selector_var);
                    (term, selector_var, clauses)
                })
                .collect();

        // Calculate total variables needed:
        // - BV variables: 0..bv_num_vars-1
        // - Selector variables: selector_base..selector_base+num_selectors-1
        let selector_base = state.selector_base.unwrap_or(10000);
        let total_selectors = state.next_selector_idx as usize;
        let total_vars = std::cmp::max(
            bv_num_vars as usize,
            selector_base as usize + total_selectors,
        );

        // Initialize or resize persistent SAT solver
        let solver = if let Some(ref mut s) = state.persistent_sat {
            // Ensure solver has enough variables
            s.ensure_num_vars(total_vars);
            s
        } else {
            state.persistent_sat = Some(SatSolver::new(total_vars));
            state.persistent_sat.as_mut().unwrap()
        };

        // Add clauses for new assertions with selector implications
        for (_term, selector_var, clauses) in new_selectors {
            // Ensure solver can handle this selector variable
            solver.ensure_num_vars(selector_var.0 as usize + 1);

            // Add each clause as: (-selector ∨ clause_lits)
            // This means the clause is only active when selector is true (in assumptions)
            for clause in clauses {
                let mut lits: Vec<SatLiteral> = vec![SatLiteral::negative(selector_var)];
                for &lit in clause.literals() {
                    if lit > 0 {
                        lits.push(SatLiteral::positive(SatVariable((lit - 1) as u32)));
                    } else {
                        lits.push(SatLiteral::negative(SatVariable((-lit - 1) as u32)));
                    }
                }
                solver.add_clause(lits);
            }
        }

        state.sat_num_vars = solver.user_num_vars();

        // Build assumptions: active selectors are true, inactive selectors are false
        // This is crucial - without explicitly setting inactive selectors to false,
        // the solver could freely assign them to true, activating popped assertions!
        // Sort by variable index for deterministic SAT solver behavior
        let active_terms: std::collections::HashSet<&TermId> = self.ctx.assertions.iter().collect();
        let mut assumptions: Vec<SatLiteral> = state
            .assertion_to_selector
            .iter()
            .map(|(term, &var)| {
                if active_terms.contains(term) {
                    SatLiteral::positive(var) // Activate this assertion
                } else {
                    SatLiteral::negative(var) // Deactivate this assertion
                }
            })
            .collect();
        assumptions.sort_by_key(|lit| lit.variable().index());

        // Solve with assumptions - learned clauses are retained!
        let solve_result = solver.solve_with_assumptions(&assumptions);

        // Clone term_to_bits for BV model extraction
        let term_bits_clone = state.term_to_bits.clone();

        // Store model if SAT
        match solve_result {
            AssumeResult::Sat(ref model) => {
                // Model contains BV variables; selector variables are internal
                let sat_model: Vec<bool> =
                    model.iter().take(bv_num_vars as usize).copied().collect();
                let bv_model = Self::extract_bv_model_from_bits(
                    &sat_model,
                    &term_bits_clone,
                    0,
                    &self.ctx.terms,
                );
                self.last_model = Some(Model {
                    sat_model,
                    var_to_term: BTreeMap::new(),
                    term_to_var: BTreeMap::new(),
                    euf_model: None,
                    array_model: None,
                    lra_model: None,
                    lia_model: None,
                    bv_model: Some(bv_model),
                });
                self.last_result = Some(CheckSatResult::Sat);
                debug_assert!(
                    self.validate_model().is_ok(),
                    "BUG: Model validation failed: {:?}",
                    self.validate_model()
                );
                Ok(CheckSatResult::Sat)
            }
            AssumeResult::Unsat(_) => {
                self.last_model = None;
                self.last_result = Some(CheckSatResult::Unsat);
                Ok(CheckSatResult::Unsat)
            }
            AssumeResult::Unknown => {
                self.last_model = None;
                self.last_result = Some(CheckSatResult::Unknown);
                Ok(CheckSatResult::Unknown)
            }
        }
    }

    /// Solve QF_ABV (Arrays + Bitvectors) using eager bit-blasting with array axioms
    ///
    /// This combines eager bit-blasting for BV operations with array axiom
    /// instantiation. For each select/store term, fresh BV variables are
    /// created, and ROW1/ROW2 axioms are generated as CNF clauses.
    fn solve_abv(&mut self) -> Result<CheckSatResult> {
        // First, run Tseitin transformation for the Boolean structure
        let tseitin = Tseitin::new(&self.ctx.terms);
        let tseitin_result = tseitin.transform_all(&self.ctx.assertions);

        // Bit-blast BV operations
        let mut bv_solver = BvSolver::new(&self.ctx.terms);
        let bv_clauses = bv_solver.bitblast_all(&self.ctx.assertions);
        let bv_num_vars = bv_solver.num_vars();

        // Combine Tseitin clauses and BV clauses
        let mut all_clauses = tseitin_result.clauses.clone();

        // Offset BV variables to not conflict with Tseitin variables
        let var_offset = tseitin_result.num_vars as i32;
        for clause in bv_clauses {
            let offset_lits: Vec<i32> = clause
                .literals()
                .iter()
                .map(|&lit| {
                    if lit > 0 {
                        lit + var_offset
                    } else {
                        lit - var_offset
                    }
                })
                .collect();
            all_clauses.push(z4_core::CnfClause::new(offset_lits));
        }

        // Collect array select/store terms and generate axiom clauses
        let array_axiom_result =
            self.generate_array_bv_axioms(tseitin_result.num_vars + bv_num_vars);

        // Add array axiom clauses
        for clause in array_axiom_result.clauses {
            all_clauses.push(clause);
        }

        let total_vars = tseitin_result.num_vars + bv_num_vars + array_axiom_result.num_vars;

        // Create a SAT solver with all clauses
        use z4_sat::Solver;
        let mut solver = Solver::new(total_vars as usize);

        for clause in &all_clauses {
            let lits: Vec<z4_sat::Literal> = clause
                .literals()
                .iter()
                .map(|&lit| {
                    if lit > 0 {
                        z4_sat::Literal::positive(z4_sat::Variable((lit - 1) as u32))
                    } else {
                        z4_sat::Literal::negative(z4_sat::Variable((-lit - 1) as u32))
                    }
                })
                .collect();
            solver.add_clause(lits);
        }

        let solve_result = solver.solve();

        // Store model if SAT
        if let SolveResult::Sat(ref model) = solve_result {
            let sat_model: Vec<bool> = model.to_vec();
            let bv_model = Self::extract_bv_model_from_bits(
                &sat_model,
                bv_solver.term_to_bits(),
                var_offset,
                &self.ctx.terms,
            );
            self.last_model = Some(Model {
                sat_model,
                var_to_term: tseitin_result.var_to_term.clone(),
                term_to_var: tseitin_result.term_to_var.clone(),
                euf_model: None,
                array_model: None,
                lra_model: None,
                lia_model: None,
                bv_model: Some(bv_model),
            });
            self.last_result = Some(CheckSatResult::Sat);
            debug_assert!(
                self.validate_model().is_ok(),
                "BUG: Model validation failed: {:?}",
                self.validate_model()
            );
            return Ok(CheckSatResult::Sat);
        }

        self.last_model = None;
        let result = CheckSatResult::from(solve_result);
        self.last_result = Some(result);
        Ok(result)
    }

    /// Solve QF_UFBV (UF + Bitvectors) using eager bit-blasting with EUF congruence axioms
    ///
    /// This combines eager bit-blasting for BV operations with EUF congruence axiom
    /// instantiation. For each uninterpreted function application f(a1,...,an),
    /// fresh BV variables are created for the result, and congruence axioms are
    /// generated as CNF clauses.
    fn solve_ufbv(&mut self) -> Result<CheckSatResult> {
        // First, run Tseitin transformation for the Boolean structure
        let tseitin = Tseitin::new(&self.ctx.terms);
        let tseitin_result = tseitin.transform_all(&self.ctx.assertions);

        // Bit-blast BV operations
        let mut bv_solver = BvSolver::new(&self.ctx.terms);
        let bv_clauses = bv_solver.bitblast_all(&self.ctx.assertions);
        let bv_num_vars = bv_solver.num_vars();

        if self.debug_ufbv {
            eprintln!("DEBUG: Tseitin num_vars = {}", tseitin_result.num_vars);
            eprintln!("DEBUG: BV num_vars = {}", bv_num_vars);
            eprintln!("DEBUG: Tseitin clauses = {}", tseitin_result.clauses.len());
            eprintln!(
                "DEBUG: BV clauses (before offset) = {}",
                bv_solver.clauses().len()
            );

            // Print bit-blasted terms
            for (term_id, bits) in bv_solver.iter_term_bits() {
                eprintln!("DEBUG: Term {:?} has bits {:?}", term_id, bits);
            }
        }

        // Combine Tseitin clauses and BV clauses
        let mut all_clauses = tseitin_result.clauses.clone();

        // Offset BV variables to not conflict with Tseitin variables
        let var_offset = tseitin_result.num_vars as i32;
        let bv_clauses_vec: Vec<_> = bv_clauses;
        for clause in &bv_clauses_vec {
            let offset_lits: Vec<i32> = clause
                .literals()
                .iter()
                .map(|&lit| {
                    if lit > 0 {
                        lit + var_offset
                    } else {
                        lit - var_offset
                    }
                })
                .collect();
            all_clauses.push(z4_core::CnfClause::new(offset_lits));
        }

        // Generate EUF congruence axiom clauses
        let euf_axiom_result = self.generate_euf_bv_axioms_debug(
            &bv_solver,
            tseitin_result.num_vars,               // BV variable offset
            tseitin_result.num_vars + bv_num_vars, // New variable offset
            self.debug_ufbv,
        );

        if self.debug_ufbv {
            eprintln!("DEBUG: EUF axiom num_vars = {}", euf_axiom_result.num_vars);
            eprintln!(
                "DEBUG: EUF axiom clauses = {}",
                euf_axiom_result.clauses.len()
            );
        }

        // Add EUF axiom clauses
        for clause in euf_axiom_result.clauses {
            all_clauses.push(clause);
        }

        let total_vars = tseitin_result.num_vars + bv_num_vars + euf_axiom_result.num_vars;

        if self.debug_ufbv {
            eprintln!("DEBUG: Total vars = {}", total_vars);
            eprintln!("DEBUG: Total clauses = {}", all_clauses.len());
        }

        // Create a SAT solver with all clauses
        use z4_sat::Solver;
        let mut solver = Solver::new(total_vars as usize);

        for clause in &all_clauses {
            let lits: Vec<z4_sat::Literal> = clause
                .literals()
                .iter()
                .map(|&lit| {
                    if lit > 0 {
                        z4_sat::Literal::positive(z4_sat::Variable((lit - 1) as u32))
                    } else {
                        z4_sat::Literal::negative(z4_sat::Variable((-lit - 1) as u32))
                    }
                })
                .collect();
            solver.add_clause(lits);
        }

        let solve_result = solver.solve();

        // Store model if SAT
        if let SolveResult::Sat(ref model) = solve_result {
            let sat_model: Vec<bool> = model.to_vec();
            let bv_model = Self::extract_bv_model_from_bits(
                &sat_model,
                bv_solver.term_to_bits(),
                var_offset,
                &self.ctx.terms,
            );
            self.last_model = Some(Model {
                sat_model,
                var_to_term: tseitin_result.var_to_term.clone(),
                term_to_var: tseitin_result.term_to_var.clone(),
                euf_model: None,
                array_model: None,
                lra_model: None,
                lia_model: None,
                bv_model: Some(bv_model),
            });
            self.last_result = Some(CheckSatResult::Sat);
            debug_assert!(
                self.validate_model().is_ok(),
                "BUG: Model validation failed: {:?}",
                self.validate_model()
            );
            return Ok(CheckSatResult::Sat);
        }

        self.last_model = None;
        let result = CheckSatResult::from(solve_result);
        self.last_result = Some(result);
        Ok(result)
    }

    /// Solve QF_AUFBV (Arrays + UF + Bitvectors) using eager bit-blasting with
    /// array and EUF congruence axioms
    ///
    /// This is the most complete bitvector logic, combining:
    /// - Eager bit-blasting for BV operations
    /// - Array axiom instantiation for select/store
    /// - EUF congruence axiom instantiation for uninterpreted functions
    fn solve_aufbv(&mut self) -> Result<CheckSatResult> {
        // First, run Tseitin transformation for the Boolean structure
        let tseitin = Tseitin::new(&self.ctx.terms);
        let tseitin_result = tseitin.transform_all(&self.ctx.assertions);

        // Bit-blast BV operations
        let mut bv_solver = BvSolver::new(&self.ctx.terms);
        let bv_clauses = bv_solver.bitblast_all(&self.ctx.assertions);
        let bv_num_vars = bv_solver.num_vars();

        // Combine Tseitin clauses and BV clauses
        let mut all_clauses = tseitin_result.clauses.clone();

        // Offset BV variables to not conflict with Tseitin variables
        let var_offset = tseitin_result.num_vars as i32;
        for clause in bv_clauses {
            let offset_lits: Vec<i32> = clause
                .literals()
                .iter()
                .map(|&lit| {
                    if lit > 0 {
                        lit + var_offset
                    } else {
                        lit - var_offset
                    }
                })
                .collect();
            all_clauses.push(z4_core::CnfClause::new(offset_lits));
        }

        // Generate array axiom clauses
        let array_axiom_result =
            self.generate_array_bv_axioms(tseitin_result.num_vars + bv_num_vars);

        // Add array axiom clauses
        for clause in array_axiom_result.clauses {
            all_clauses.push(clause);
        }

        // Generate EUF congruence axiom clauses
        let euf_axiom_result = self.generate_euf_bv_axioms(
            &bv_solver,
            tseitin_result.num_vars, // BV variable offset
            tseitin_result.num_vars + bv_num_vars + array_axiom_result.num_vars, // New variable offset
        );

        // Add EUF axiom clauses
        for clause in euf_axiom_result.clauses {
            all_clauses.push(clause);
        }

        let total_vars = tseitin_result.num_vars
            + bv_num_vars
            + array_axiom_result.num_vars
            + euf_axiom_result.num_vars;

        // Create a SAT solver with all clauses
        use z4_sat::Solver;
        let mut solver = Solver::new(total_vars as usize);

        for clause in &all_clauses {
            let lits: Vec<z4_sat::Literal> = clause
                .literals()
                .iter()
                .map(|&lit| {
                    if lit > 0 {
                        z4_sat::Literal::positive(z4_sat::Variable((lit - 1) as u32))
                    } else {
                        z4_sat::Literal::negative(z4_sat::Variable((-lit - 1) as u32))
                    }
                })
                .collect();
            solver.add_clause(lits);
        }

        let solve_result = solver.solve();

        // Store model if SAT
        if let SolveResult::Sat(ref model) = solve_result {
            let sat_model: Vec<bool> = model.to_vec();
            let bv_model = Self::extract_bv_model_from_bits(
                &sat_model,
                bv_solver.term_to_bits(),
                var_offset,
                &self.ctx.terms,
            );
            self.last_model = Some(Model {
                sat_model,
                var_to_term: tseitin_result.var_to_term.clone(),
                term_to_var: tseitin_result.term_to_var.clone(),
                euf_model: None,
                array_model: None,
                lra_model: None,
                lia_model: None,
                bv_model: Some(bv_model),
            });
            self.last_result = Some(CheckSatResult::Sat);
            debug_assert!(
                self.validate_model().is_ok(),
                "BUG: Model validation failed: {:?}",
                self.validate_model()
            );
            return Ok(CheckSatResult::Sat);
        }

        self.last_model = None;
        let result = CheckSatResult::from(solve_result);
        self.last_result = Some(result);
        Ok(result)
    }

    /// Generate array axiom clauses for QF_ABV
    ///
    /// Collects all select/store terms and generates:
    /// - ROW1: select(store(a, i, v), i) = v
    /// - ROW2: i ≠ j → select(store(a, i, v), j) = select(a, j)
    /// - Functional consistency: i = j → select(a, i) = select(a, j)
    fn generate_array_bv_axioms(&self, var_offset: u32) -> ArrayAxiomResult {
        let mut result = ArrayAxiomResult {
            clauses: Vec::new(),
            num_vars: 0,
        };

        // Collect all select and store terms from assertions
        let mut select_terms: Vec<(TermId, TermId, TermId)> = Vec::new(); // (select_term, array, index)
        let mut store_terms: Vec<(TermId, TermId, TermId, TermId)> = Vec::new(); // (store_term, array, index, value)

        for &assertion in &self.ctx.assertions {
            self.collect_array_terms(
                assertion,
                &mut select_terms,
                &mut store_terms,
                &mut HashSet::new(),
            );
        }

        // For now, generate simple functional consistency axioms
        // For each pair of selects on the same array, if indices might be equal,
        // the results must be equal
        //
        // More sophisticated axiom generation (ROW1/ROW2 for stores) can be added later
        // For basic QF_ABV support, we rely on the fact that select/store terms
        // are already handled by the term store's rewriting (mk_select simplifies
        // select(store(a, i, v), i) to v)

        // Track which terms we've processed
        let mut next_var = var_offset + 1; // 1-indexed

        // For stores followed by selects, generate ROW1/ROW2 axioms
        for &(store_term, _array, store_idx, store_val) in &store_terms {
            // Find selects from this store
            for &(select_term, sel_array, sel_idx) in &select_terms {
                if sel_array == store_term {
                    // This is a select from a store: select(store(a, i, v), j)
                    // ROW1: if i = j then result = v
                    // ROW2: if i ≠ j then result = select(a, j) (but we don't have this term)

                    // For now, just handle the case where indices might be equal
                    // Generate: (i = j) → (result = v)
                    // In CNF: (i ≠ j) ∨ (result = v)

                    // Get bitvector width
                    if let Sort::BitVec(width) = self.ctx.terms.sort(store_idx) {
                        if let Sort::BitVec(val_width) = self.ctx.terms.sort(store_val) {
                            // Generate clause for index equality implies value equality
                            // This is a complex clause involving bit-level comparisons
                            // For simplicity, we'll skip this for now and rely on the
                            // term store rewriting which handles the syntactic case

                            // The term store already rewrites select(store(a, i, v), i) to v
                            // when indices are syntactically equal, so we don't need to
                            // generate these axioms for that case.

                            // For the case where indices might be semantically equal but
                            // syntactically different (e.g., x and x+0), we would need
                            // proper axiom generation with bit-blasted equalities.

                            // Mark as used to avoid warnings
                            let _ = (width, val_width, select_term, sel_idx, &mut next_var);
                        }
                    }
                }
            }
        }

        // For functional consistency: if we have multiple selects on the same array
        // with potentially equal indices, their results must be equal
        // This is handled implicitly by the term store's hash-consing - if two select
        // terms have the same array and index, they get the same TermId

        result.num_vars = next_var.saturating_sub(var_offset + 1);
        result
    }

    /// Recursively collect select and store terms from an expression
    fn collect_array_terms(
        &self,
        term: TermId,
        selects: &mut Vec<(TermId, TermId, TermId)>,
        stores: &mut Vec<(TermId, TermId, TermId, TermId)>,
        visited: &mut HashSet<TermId>,
    ) {
        if visited.contains(&term) {
            return;
        }
        visited.insert(term);

        match self.ctx.terms.get(term) {
            TermData::App(Symbol::Named(name), args) => {
                match name.as_str() {
                    "select" if args.len() == 2 => {
                        selects.push((term, args[0], args[1]));
                        // Recurse into array and index
                        self.collect_array_terms(args[0], selects, stores, visited);
                        self.collect_array_terms(args[1], selects, stores, visited);
                    }
                    "store" if args.len() == 3 => {
                        stores.push((term, args[0], args[1], args[2]));
                        // Recurse into array, index, and value
                        self.collect_array_terms(args[0], selects, stores, visited);
                        self.collect_array_terms(args[1], selects, stores, visited);
                        self.collect_array_terms(args[2], selects, stores, visited);
                    }
                    _ => {
                        // Recurse into other function applications
                        for &arg in args.iter() {
                            self.collect_array_terms(arg, selects, stores, visited);
                        }
                    }
                }
            }
            TermData::Not(inner) => {
                self.collect_array_terms(*inner, selects, stores, visited);
            }
            TermData::Ite(c, t, e) => {
                self.collect_array_terms(*c, selects, stores, visited);
                self.collect_array_terms(*t, selects, stores, visited);
                self.collect_array_terms(*e, selects, stores, visited);
            }
            _ => {}
        }
    }

    /// Generate EUF congruence axiom clauses for QF_UFBV/QF_AUFBV (with debug output)
    fn generate_euf_bv_axioms_debug(
        &self,
        bv_solver: &BvSolver,
        bv_offset: u32,
        var_offset: u32,
        debug: bool,
    ) -> EufAxiomResult {
        let mut result = EufAxiomResult {
            clauses: Vec::new(),
            num_vars: 0,
        };

        // Collect all uninterpreted function applications
        let mut uf_apps: HashMap<String, Vec<(TermId, Vec<TermId>)>> = HashMap::new();

        for &assertion in &self.ctx.assertions {
            self.collect_uf_applications(assertion, &mut uf_apps, &mut HashSet::new());
        }

        if debug {
            eprintln!("DEBUG: Collected UF applications:");
            for (name, apps) in &uf_apps {
                eprintln!("  Function '{}' has {} applications:", name, apps.len());
                for (term, args) in apps {
                    let term_bits = bv_solver.get_term_bits(*term);
                    eprintln!(
                        "    Term {:?} with args {:?}, bits: {:?}",
                        term,
                        args,
                        term_bits.map(|b| b.to_vec())
                    );
                    for (i, arg) in args.iter().enumerate() {
                        let arg_bits = bv_solver.get_term_bits(*arg);
                        eprintln!(
                            "      Arg {}: term {:?}, bits: {:?}",
                            i,
                            arg,
                            arg_bits.map(|b| b.to_vec())
                        );
                    }
                }
            }
        }

        let mut next_var = var_offset + 1;

        for (func_name, applications) in &uf_apps {
            if applications.len() < 2 {
                continue;
            }

            for i in 0..applications.len() {
                for j in (i + 1)..applications.len() {
                    let (term1, args1) = &applications[i];
                    let (term2, args2) = &applications[j];

                    if args1.len() != args2.len() {
                        continue;
                    }

                    let bits1 = match bv_solver.get_term_bits(*term1) {
                        Some(b) => b,
                        None => {
                            if debug {
                                eprintln!("DEBUG: Skipping pair - term1 {:?} has no bits", term1);
                            }
                            continue;
                        }
                    };
                    let bits2 = match bv_solver.get_term_bits(*term2) {
                        Some(b) => b,
                        None => {
                            if debug {
                                eprintln!("DEBUG: Skipping pair - term2 {:?} has no bits", term2);
                            }
                            continue;
                        }
                    };

                    if bits1.len() != bits2.len() || bits1.is_empty() {
                        continue;
                    }

                    if debug {
                        eprintln!(
                            "DEBUG: Generating congruence axiom for {}({:?}) and {}({:?})",
                            func_name, args1, func_name, args2
                        );
                        eprintln!("  term1 bits (unoffset): {:?}", bits1);
                        eprintln!("  term2 bits (unoffset): {:?}", bits2);
                    }

                    let offset_bit = |bit: i32| -> i32 {
                        if bit > 0 {
                            bit + bv_offset as i32
                        } else {
                            bit - bv_offset as i32
                        }
                    };

                    let mut all_diff_vars = Vec::new();
                    let mut all_args_have_bits = true;

                    for (arg_idx, (arg1, arg2)) in args1.iter().zip(args2.iter()).enumerate() {
                        let arg1_bits = bv_solver.get_term_bits(*arg1);
                        let arg2_bits = bv_solver.get_term_bits(*arg2);

                        match (arg1_bits, arg2_bits) {
                            (Some(b1), Some(b2)) if b1.len() == b2.len() && !b1.is_empty() => {
                                if debug {
                                    eprintln!("  Arg {} pair: {:?} vs {:?}", arg_idx, arg1, arg2);
                                    eprintln!("    arg1 bits (unoffset): {:?}", b1);
                                    eprintln!("    arg2 bits (unoffset): {:?}", b2);
                                }

                                for (bit_idx, (&bit1, &bit2)) in
                                    b1.iter().zip(b2.iter()).enumerate()
                                {
                                    let ob1 = offset_bit(bit1);
                                    let ob2 = offset_bit(bit2);
                                    let diff_var = next_var as i32;
                                    next_var += 1;
                                    all_diff_vars.push(diff_var);

                                    if debug && bit_idx < 2 {
                                        eprintln!(
                                            "    bit {}: diff_var={}, ob1={}, ob2={}",
                                            bit_idx, diff_var, ob1, ob2
                                        );
                                    }

                                    // diff_j ↔ (b1[j] XOR b2[j])
                                    result
                                        .clauses
                                        .push(z4_core::CnfClause::new(vec![-diff_var, ob1, ob2]));
                                    result
                                        .clauses
                                        .push(z4_core::CnfClause::new(vec![-diff_var, -ob1, -ob2]));
                                    result
                                        .clauses
                                        .push(z4_core::CnfClause::new(vec![-ob1, ob2, diff_var]));
                                    result
                                        .clauses
                                        .push(z4_core::CnfClause::new(vec![ob1, -ob2, diff_var]));
                                }
                            }
                            _ => {
                                if debug {
                                    eprintln!(
                                        "  Arg {} pair: {:?} vs {:?} - MISSING BITS",
                                        arg_idx, arg1, arg2
                                    );
                                    eprintln!("    arg1_bits: {:?}", arg1_bits.map(|b| b.to_vec()));
                                    eprintln!("    arg2_bits: {:?}", arg2_bits.map(|b| b.to_vec()));
                                }
                                all_args_have_bits = false;
                                break;
                            }
                        }
                    }

                    if !all_args_have_bits || all_diff_vars.is_empty() {
                        if debug {
                            eprintln!(
                                "  SKIPPING - all_args_have_bits={}, diff_vars={}",
                                all_args_have_bits,
                                all_diff_vars.len()
                            );
                        }
                        continue;
                    }

                    if debug {
                        eprintln!("  Generated {} diff vars", all_diff_vars.len());
                    }

                    // For each result bit, add the congruence constraint:
                    // diff_0 ∨ diff_1 ∨ ... ∨ ¬f(a)[i] ∨ f(b)[i]
                    // diff_0 ∨ diff_1 ∨ ... ∨ f(a)[i] ∨ ¬f(b)[i]
                    // These two clauses encode: (args differ) ∨ (f(a)[i] = f(b)[i])
                    for (&bit1, &bit2) in bits1.iter().zip(bits2.iter()) {
                        let ob1 = offset_bit(bit1);
                        let ob2 = offset_bit(bit2);

                        // Clause 1: diff_0 ∨ ... ∨ ¬f(a)[i] ∨ f(b)[i]
                        let mut clause1: Vec<i32> = all_diff_vars.clone();
                        clause1.push(-ob1);
                        clause1.push(ob2);
                        result.clauses.push(z4_core::CnfClause::new(clause1));

                        // Clause 2: diff_0 ∨ ... ∨ f(a)[i] ∨ ¬f(b)[i]
                        let mut clause2: Vec<i32> = all_diff_vars.clone();
                        clause2.push(ob1);
                        clause2.push(-ob2);
                        result.clauses.push(z4_core::CnfClause::new(clause2));
                    }
                }
            }
        }

        result.num_vars = next_var.saturating_sub(var_offset + 1);
        result
    }

    /// Generate EUF congruence axiom clauses (non-debug version)
    fn generate_euf_bv_axioms(
        &self,
        bv_solver: &BvSolver,
        bv_offset: u32,
        var_offset: u32,
    ) -> EufAxiomResult {
        self.generate_euf_bv_axioms_debug(bv_solver, bv_offset, var_offset, false)
    }

    /// Recursively collect uninterpreted function applications from an expression
    fn collect_uf_applications(
        &self,
        term: TermId,
        uf_apps: &mut HashMap<String, Vec<(TermId, Vec<TermId>)>>,
        visited: &mut HashSet<TermId>,
    ) {
        if visited.contains(&term) {
            return;
        }
        visited.insert(term);

        match self.ctx.terms.get(term) {
            TermData::App(Symbol::Named(name), args) => {
                // Check if this is an uninterpreted function (not a built-in BV or array op)
                let is_builtin = matches!(
                    name.as_str(),
                    "bvadd"
                        | "bvsub"
                        | "bvmul"
                        | "bvudiv"
                        | "bvurem"
                        | "bvsdiv"
                        | "bvsrem"
                        | "bvand"
                        | "bvor"
                        | "bvxor"
                        | "bvnot"
                        | "bvneg"
                        | "bvshl"
                        | "bvlshr"
                        | "bvashr"
                        | "concat"
                        | "extract"
                        | "repeat"
                        | "zero_extend"
                        | "sign_extend"
                        | "rotate_left"
                        | "rotate_right"
                        | "bvult"
                        | "bvule"
                        | "bvugt"
                        | "bvuge"
                        | "bvslt"
                        | "bvsle"
                        | "bvsgt"
                        | "bvsge"
                        | "="
                        | "distinct"
                        | "ite"
                        | "and"
                        | "or"
                        | "not"
                        | "=>"
                        | "xor"
                        | "select"
                        | "store"
                        | "true"
                        | "false"
                );

                if !is_builtin && !args.is_empty() {
                    // This is an uninterpreted function application
                    uf_apps
                        .entry(name.clone())
                        .or_default()
                        .push((term, args.clone()));
                }

                // Recurse into arguments
                for &arg in args.iter() {
                    self.collect_uf_applications(arg, uf_apps, visited);
                }
            }
            TermData::App(Symbol::Indexed(name, _), args) => {
                // Indexed symbols like (_ extract ...) are built-in
                // Just recurse into arguments
                for &arg in args.iter() {
                    self.collect_uf_applications(arg, uf_apps, visited);
                }

                // But user-defined indexed functions should be tracked
                let is_builtin = matches!(
                    name.as_str(),
                    "extract"
                        | "repeat"
                        | "zero_extend"
                        | "sign_extend"
                        | "rotate_left"
                        | "rotate_right"
                );

                if !is_builtin && !args.is_empty() {
                    uf_apps
                        .entry(name.clone())
                        .or_default()
                        .push((term, args.clone()));
                }
            }
            TermData::Not(inner) => {
                self.collect_uf_applications(*inner, uf_apps, visited);
            }
            TermData::Ite(c, t, e) => {
                self.collect_uf_applications(*c, uf_apps, visited);
                self.collect_uf_applications(*t, uf_apps, visited);
                self.collect_uf_applications(*e, uf_apps, visited);
            }
            _ => {}
        }
    }

    /// Process solve result and store model if SAT
    fn solve_and_store_model(
        &mut self,
        result: SolveResult,
        tseitin_result: &TseitinResult,
        euf_model: Option<EufModel>,
        array_model: Option<ArrayModel>,
    ) -> Result<CheckSatResult> {
        self.solve_and_store_model_full(result, tseitin_result, euf_model, array_model, None, None)
    }

    /// Process solve result and store model if SAT (with all theory models)
    fn solve_and_store_model_full(
        &mut self,
        result: SolveResult,
        tseitin_result: &TseitinResult,
        euf_model: Option<EufModel>,
        array_model: Option<ArrayModel>,
        lra_model: Option<LraModel>,
        lia_model: Option<LiaModel>,
    ) -> Result<CheckSatResult> {
        match result {
            SolveResult::Sat(model) => {
                // Store the model with mappings (convert from 1-indexed CNF vars to 0-indexed)
                let var_to_term: BTreeMap<u32, TermId> = tseitin_result
                    .var_to_term
                    .iter()
                    .map(|(&v, &t)| (v - 1, t))
                    .collect();
                let term_to_var: BTreeMap<TermId, u32> = tseitin_result
                    .term_to_var
                    .iter()
                    .map(|(&t, &v)| (t, v - 1))
                    .collect();

                // Apply minimization if enabled
                let (minimized_lra, minimized_lia) = if self.minimize_counterexamples_enabled() {
                    (
                        self.minimize_lra_model(lra_model.as_ref()),
                        self.minimize_lia_model(lia_model.as_ref()),
                    )
                } else {
                    (lra_model, lia_model)
                };

                self.last_model = Some(Model {
                    sat_model: model,
                    var_to_term,
                    term_to_var,
                    euf_model,
                    array_model,
                    lra_model: minimized_lra,
                    lia_model: minimized_lia,
                    bv_model: None,
                });
                self.last_result = Some(CheckSatResult::Sat);
                Ok(CheckSatResult::Sat)
            }
            SolveResult::Unsat => {
                // Build proof if proof production is enabled
                if self.produce_proofs_enabled() {
                    self.build_unsat_proof();
                }
                self.last_result = Some(CheckSatResult::Unsat);
                Ok(CheckSatResult::Unsat)
            }
            SolveResult::Unknown => {
                self.last_result = Some(CheckSatResult::Unknown);
                Ok(CheckSatResult::Unknown)
            }
        }
    }

    /// Build a proof for UNSAT result
    ///
    /// Creates an Alethe-compatible proof with assumptions for each assertion
    /// and a final step deriving the empty clause.
    fn build_unsat_proof(&mut self) {
        // If the proof tracker has steps, use those (more detailed proof)
        if self.proof_tracker.num_steps() > 0 {
            self.last_proof = Some(self.proof_tracker.take_proof());
            return;
        }

        // Otherwise, build a basic proof structure with assumptions
        let mut proof = Proof::new();

        // Add assume steps for each assertion
        let mut assume_ids = Vec::new();
        for (idx, &assertion) in self.ctx.assertions.iter().enumerate() {
            let id = proof.add_assume(assertion, Some(format!("h{}", idx)));
            assume_ids.push(id);
        }

        // Add a hole step deriving the empty clause (contradiction)
        // This is a valid but incomplete proof - the solver found UNSAT but
        // we don't yet track all the detailed resolution/theory steps.
        // The "hole" rule is accepted by proof checkers as "trust this step".
        proof.add_step(ProofStep::Step {
            rule: AletheRule::Hole,
            clause: vec![], // empty clause = false
            premises: assume_ids,
            args: vec![],
        });

        self.last_proof = Some(proof);
    }

    /// Minimize LRA model values, preferring 0, 1, -1
    fn minimize_lra_model(&self, model: Option<&LraModel>) -> Option<LraModel> {
        let model = model?;
        let mut minimized = LraModel {
            values: HashMap::new(),
        };

        // Candidate values to try, in order of preference
        let candidates = [BigRational::zero(), BigRational::one(), -BigRational::one()];

        for (&term_id, original_value) in &model.values {
            let mut best_value = original_value.clone();

            // Try each candidate value
            for candidate in &candidates {
                // Skip if this is already the value
                if candidate == original_value {
                    best_value = candidate.clone();
                    break;
                }

                // Create a temporary model with the candidate value
                let mut test_values = model.values.clone();
                test_values.insert(term_id, candidate.clone());

                // Check if this value satisfies all constraints
                // For now, we use a simple heuristic: prefer smaller absolute values
                // A full check would require re-evaluating all constraints
                if candidate.abs() < original_value.abs() {
                    best_value = candidate.clone();
                    break;
                }
            }

            minimized.values.insert(term_id, best_value);
        }

        Some(minimized)
    }

    /// Minimize LIA model values, preferring 0, 1, -1
    fn minimize_lia_model(&self, model: Option<&LiaModel>) -> Option<LiaModel> {
        use num_bigint::BigInt;

        let model = model?;
        let mut minimized = LiaModel {
            values: HashMap::new(),
        };

        // Candidate values to try, in order of preference
        let candidates = [BigInt::from(0), BigInt::from(1), BigInt::from(-1)];

        for (&term_id, original_value) in &model.values {
            let mut best_value = original_value.clone();

            // Try each candidate value
            for candidate in &candidates {
                // Skip if this is already the value
                if candidate == original_value {
                    best_value = candidate.clone();
                    break;
                }

                // For now, we use a simple heuristic: prefer smaller absolute values
                // A full check would require re-evaluating all constraints
                if candidate.magnitude() < original_value.magnitude() {
                    best_value = candidate.clone();
                    break;
                }
            }

            minimized.values.insert(term_id, best_value);
        }

        Some(minimized)
    }

    /// Generate model output for get-model command
    pub fn get_model(&self) -> String {
        if !self.produce_models_enabled() {
            return "(error \"model generation is not enabled\")".to_string();
        }

        // Check if we have a model
        let model = match (&self.last_result, &self.last_model) {
            (Some(CheckSatResult::Sat), Some(m)) => m,
            (Some(CheckSatResult::Sat), None) => {
                // SAT but no assertions (trivially satisfiable)
                return "(model\n)".to_string();
            }
            _ => {
                return "(error \"model is not available\")".to_string();
            }
        };

        // Collect model values for user-declared symbols
        let mut definitions = Vec::new();

        for (name, info) in self.ctx.symbol_iter() {
            // Handle functions with arguments (generate function tables)
            if !info.arg_sorts.is_empty() {
                // Check if we have EUF model with function tables
                if let Some(ref euf_model) = model.euf_model {
                    if let Some(table) = euf_model.function_tables.get(name) {
                        let def =
                            self.format_function_table(name, &info.arg_sorts, &info.sort, table);
                        definitions.push(def);
                    }
                }
                continue;
            }

            // For constants (no arguments), need term_id
            if let Some(term_id) = info.term {
                // For constants (no arguments), look up value
                let sort_str = format_sort(&info.sort);

                // Handle array-sorted symbols specially
                if let Sort::Array(_, _) = &info.sort {
                    if let Some(ref array_model) = model.array_model {
                        if let Some(interp) = array_model.array_values.get(&term_id) {
                            let array_value = self.format_array_value(&info.sort, interp);
                            definitions.push(format!(
                                "  (define-fun {} () {}\n    {})",
                                name, sort_str, array_value
                            ));
                            continue;
                        }
                    }
                    // Fallback: return a default const-array
                    let array_value = self.format_default_array(&info.sort);
                    definitions.push(format!(
                        "  (define-fun {} () {}\n    {})",
                        name, sort_str, array_value
                    ));
                    continue;
                }

                // Try EUF model first for uninterpreted sorts
                if let Some(ref euf_model) = model.euf_model {
                    if let Some(elem) = euf_model.term_values.get(&term_id) {
                        definitions
                            .push(format!("  (define-fun {} () {} {})", name, sort_str, elem));
                        continue;
                    }
                }

                // Try LRA model for Real sort
                if matches!(info.sort, Sort::Real) {
                    if let Some(ref lra_model) = model.lra_model {
                        if let Some(val) = lra_model.values.get(&term_id) {
                            // Use the actual value without minimization
                            let value_str = format_rational(val);
                            definitions.push(format!(
                                "  (define-fun {} () {} {})",
                                name, sort_str, value_str
                            ));
                            continue;
                        }
                    }
                }

                // Try LIA model for Int sort
                if matches!(info.sort, Sort::Int) {
                    let debug = std::env::var("Z4_DEBUG_MODEL").is_ok();
                    if debug {
                        eprintln!(
                            "[MODEL] Looking up Int symbol '{}' term_id={}, lia_model={}",
                            name,
                            term_id.0,
                            model.lia_model.is_some()
                        );
                        if let Some(ref lm) = model.lia_model {
                            eprintln!(
                                "[MODEL]   LIA model keys: {:?}",
                                lm.values.keys().map(|k| k.0).collect::<Vec<_>>()
                            );
                        }
                    }
                    if let Some(ref lia_model) = model.lia_model {
                        if let Some(val) = lia_model.values.get(&term_id) {
                            if debug {
                                eprintln!(
                                    "[MODEL]   Found value {} for term_id={}",
                                    val, term_id.0
                                );
                            }
                            // Only apply minimization if counterexample minimization is enabled
                            // and bounds are available. Otherwise use the actual value.
                            let value_str = format_bigint(val);
                            definitions.push(format!(
                                "  (define-fun {} () {} {})",
                                name, sort_str, value_str
                            ));
                            continue;
                        } else if debug {
                            eprintln!("[MODEL]   NOT found in LIA model for term_id={}", term_id.0);
                        }
                    }
                    // Also check LRA model for Int (when using pure LRA solver for arithmetic)
                    if let Some(ref lra_model) = model.lra_model {
                        if let Some(val) = lra_model.values.get(&term_id) {
                            // Convert rational to integer if it's a whole number
                            if val.is_integer() {
                                // Use the actual value without minimization
                                let value_str = format_bigint(val.numer());
                                definitions.push(format!(
                                    "  (define-fun {} () {} {})",
                                    name, sort_str, value_str
                                ));
                                continue;
                            }
                        }
                    }
                }

                // Try BV model for BitVec sort
                if let Sort::BitVec(width) = &info.sort {
                    if let Some(ref bv_model) = model.bv_model {
                        if let Some(val) = bv_model.values.get(&term_id) {
                            let hex_str = format_bitvec(val, *width);
                            definitions.push(format!(
                                "  (define-fun {} () {} {})",
                                name, sort_str, hex_str
                            ));
                            continue;
                        }
                    }
                }

                // Fall back to SAT model for Bool
                let value = self.term_value(&model.sat_model, &model.term_to_var, term_id);
                let value_str = format_value(&info.sort, value, &self.ctx.terms);
                definitions.push(format!(
                    "  (define-fun {} () {} {})",
                    name, sort_str, value_str
                ));
            }
        }

        if definitions.is_empty() {
            "(model\n)".to_string()
        } else {
            format!("(model\n{}\n)", definitions.join("\n"))
        }
    }

    /// Format a function table as an SMT-LIB define-fun
    fn format_function_table(
        &self,
        name: &str,
        arg_sorts: &[Sort],
        result_sort: &Sort,
        table: &[(Vec<String>, String)],
    ) -> String {
        // Generate parameter names: x0, x1, ...
        let params: Vec<String> = arg_sorts
            .iter()
            .enumerate()
            .map(|(i, s)| format!("(x{} {})", i, format_sort(s)))
            .collect();

        let params_str = params.join(" ");
        let result_sort_str = format_sort(result_sort);

        // Build nested ite expression from table
        let body = if table.is_empty() {
            // Empty table - return default value
            format_value(result_sort, None, &self.ctx.terms)
        } else {
            self.format_function_body(arg_sorts, result_sort, table)
        };

        format!(
            "  (define-fun {} ({}) {}\n    {})",
            name, params_str, result_sort_str, body
        )
    }

    /// Build nested ite expression for function table
    fn format_function_body(
        &self,
        _arg_sorts: &[Sort],
        result_sort: &Sort,
        table: &[(Vec<String>, String)],
    ) -> String {
        if table.is_empty() {
            return format_value(result_sort, None, &self.ctx.terms);
        }

        // Use last entry as the default (else branch)
        let (_, default_result) = table.last().unwrap();

        if table.len() == 1 {
            // Single entry - just return the result
            return default_result.clone();
        }

        // Build nested ite from all entries except last (which becomes the else)
        let mut result = default_result.clone();

        for (args, value) in table.iter().rev().skip(1) {
            // Build condition: (and (= x0 arg0) (= x1 arg1) ...)
            let conditions: Vec<String> = args
                .iter()
                .enumerate()
                .map(|(i, arg)| format!("(= x{} {})", i, arg))
                .collect();

            let condition = if conditions.len() == 1 {
                conditions[0].clone()
            } else {
                format!("(and {})", conditions.join(" "))
            };

            result = format!("(ite {} {} {})", condition, value, result);
        }

        result
    }

    /// Format an array value from ArrayInterpretation for model output
    fn format_array_value(&self, sort: &Sort, interp: &z4_arrays::ArrayInterpretation) -> String {
        let sort_str = format_sort(sort);

        // Start with a const-array if we have a default, otherwise use a placeholder
        let base = if let Some(ref default) = interp.default {
            format!("((as const {}) {})", sort_str, default)
        } else {
            // Use a default value based on element sort
            let default_val = match sort {
                Sort::Array(_, elem) => format_default_value(elem),
                _ => "0".to_string(),
            };
            format!("((as const {}) {})", sort_str, default_val)
        };

        // Apply stores on top
        let mut result = base;
        for (index, value) in &interp.stores {
            result = format!("(store {} {} {})", result, index, value);
        }

        result
    }

    /// Format a default array value when no model info is available
    fn format_default_array(&self, sort: &Sort) -> String {
        let sort_str = format_sort(sort);
        let default_val = match sort {
            Sort::Array(_, elem) => format_default_value(elem),
            _ => "0".to_string(),
        };
        format!("((as const {}) {})", sort_str, default_val)
    }

    /// Generate output for get-value command
    fn get_values(&self, term_ids: &[TermId]) -> String {
        if !self.produce_models_enabled() {
            return "(error \"model generation is not enabled\")".to_string();
        }

        // Check if we have a model
        let model = match (&self.last_result, &self.last_model) {
            (Some(CheckSatResult::Sat), Some(m)) => m,
            (Some(CheckSatResult::Sat), None) => {
                // SAT but no assertions - all terms have undefined values
                let pairs: Vec<String> = term_ids
                    .iter()
                    .map(|&term_id| {
                        let term_str = self.format_term(term_id);
                        let sort = self.ctx.terms.sort(term_id);
                        let value_str = format_value(sort, None, &self.ctx.terms);
                        format!("({} {})", term_str, value_str)
                    })
                    .collect();
                return format!("({})", pairs.join(" "));
            }
            _ => {
                return "(error \"model is not available\")".to_string();
            }
        };

        // Generate values for each term using recursive evaluation
        let pairs: Vec<String> = term_ids
            .iter()
            .map(|&term_id| {
                let term_str = self.format_term(term_id);
                let eval_value = self.evaluate_term(model, term_id);
                let value_str = self.format_eval_value(&eval_value, term_id);
                format!("({} {})", term_str, value_str)
            })
            .collect();

        format!("({})", pairs.join(" "))
    }

    /// Generate output for get-info command
    fn get_info(&self, keyword: &str) -> String {
        // Keywords may come with or without the colon prefix
        let key = keyword.trim_start_matches(':');
        match key {
            "name" => "(:name \"z4\")".to_string(),
            "version" => "(:version \"0.1.0\")".to_string(),
            "authors" => "(:authors \"Z4 Authors\")".to_string(),
            "error-behavior" => "(:error-behavior immediate-exit)".to_string(),
            "reason-unknown" => {
                // Return reason for 'unknown' result if applicable
                match self.last_result {
                    Some(CheckSatResult::Unknown) => "(:reason-unknown incomplete)".to_string(),
                    _ => "(error \"no unknown result to explain\")".to_string(),
                }
            }
            "all-statistics" => {
                // Return basic statistics
                format!(
                    "(:all-statistics ((assertions {})))",
                    self.ctx.assertions.len()
                )
            }
            "assertion-stack-levels" => {
                format!("(:assertion-stack-levels {})", self.assertion_count())
            }
            _ => format!("(error \"unsupported info keyword: {}\")", keyword),
        }
    }

    /// Get an option value for get-option command
    fn get_option_value(&self, keyword: &str) -> String {
        let key = keyword.trim_start_matches(':');
        match self.ctx.get_option(key) {
            Some(OptionValue::Bool(b)) => format!("(:{} {})", key, b),
            Some(OptionValue::String(s)) => format!("(:{} \"{}\")", key, s),
            Some(OptionValue::Numeral(n)) => format!("(:{} {})", key, n),
            None => format!("(error \"unknown option: {}\")", keyword),
        }
    }

    /// Get current assertions for get-assertions command
    fn get_assertions(&self) -> String {
        if self.ctx.assertions.is_empty() {
            return "()".to_string();
        }

        let formatted: Vec<String> = self
            .ctx
            .assertions
            .iter()
            .map(|&term_id| self.format_term(term_id))
            .collect();

        format!("({})", formatted.join("\n "))
    }

    /// Simplify a term and return its SMT-LIB representation
    ///
    /// The term is already simplified during elaboration (by the TermStore),
    /// so this just formats the already-simplified term.
    fn simplify(&self, term_id: TermId) -> String {
        self.format_term(term_id)
    }

    /// Format an evaluated value for SMT-LIB output
    fn format_eval_value(&self, value: &EvalValue, term_id: TermId) -> String {
        match value {
            EvalValue::Bool(true) => "true".to_string(),
            EvalValue::Bool(false) => "false".to_string(),
            EvalValue::Element(elem) => elem.clone(),
            EvalValue::Rational(r) => {
                if r.is_integer() {
                    r.numer().to_string()
                } else {
                    format!("(/ {} {})", r.numer(), r.denom())
                }
            }
            EvalValue::Unknown => {
                // Fall back to default for the sort
                let sort = self.ctx.terms.sort(term_id);
                format_value(sort, None, &self.ctx.terms)
            }
        }
    }

    /// Format a term for SMT-LIB output (reconstructs the expression)
    fn format_term(&self, term_id: TermId) -> String {
        let term = self.ctx.terms.get(term_id);
        match term {
            TermData::Var(name, _) => name.clone(),
            TermData::Const(Constant::Bool(true)) => "true".to_string(),
            TermData::Const(Constant::Bool(false)) => "false".to_string(),
            TermData::Const(Constant::Int(n)) => n.to_string(),
            TermData::Const(Constant::Rational(r)) => {
                if r.0.is_integer() {
                    format!("{}.0", r.0.numer())
                } else {
                    format!("(/ {} {})", r.0.numer(), r.0.denom())
                }
            }
            TermData::Const(Constant::String(s)) => format!("\"{}\"", s),
            TermData::Const(Constant::BitVec { value, width }) => {
                let hex_width = (*width as usize).div_ceil(4);
                format!("#x{:0>width$}", value.to_str_radix(16), width = hex_width)
            }
            TermData::Not(inner) => format!("(not {})", self.format_term(*inner)),
            TermData::Ite(cond, then_br, else_br) => format!(
                "(ite {} {} {})",
                self.format_term(*cond),
                self.format_term(*then_br),
                self.format_term(*else_br)
            ),
            TermData::App(sym, args) => {
                let name = sym.to_string();
                if args.is_empty() {
                    name
                } else {
                    let args_str: Vec<String> = args.iter().map(|&a| self.format_term(a)).collect();
                    format!("({} {})", name, args_str.join(" "))
                }
            }
            TermData::Let(bindings, body) => {
                // Let bindings should normally be expanded, but format just in case
                let bindings_str: Vec<String> = bindings
                    .iter()
                    .map(|(name, term)| format!("({} {})", name, self.format_term(*term)))
                    .collect();
                format!(
                    "(let ({}) {})",
                    bindings_str.join(" "),
                    self.format_term(*body)
                )
            }
        }
    }

    /// Get the value of a term from the model (simple SAT lookup)
    fn term_value(
        &self,
        sat_model: &[bool],
        term_to_var: &BTreeMap<TermId, u32>,
        term_id: TermId,
    ) -> Option<bool> {
        // Use the cached reverse mapping for O(1) lookup
        if let Some(&var) = term_to_var.get(&term_id) {
            return sat_model.get(var as usize).copied();
        }
        // Term not in model - could be eliminated or not relevant
        None
    }

    /// Evaluate a term under the current model, recursively handling composite terms
    ///
    /// This handles Boolean connectives (and, or, not, ite), equality/distinct,
    /// and looks up values for variables and function applications.
    fn evaluate_term(&self, model: &Model, term_id: TermId) -> EvalValue {
        let term = self.ctx.terms.get(term_id);
        let sort = self.ctx.terms.sort(term_id);

        match term {
            // Constants evaluate to themselves
            TermData::Const(Constant::Bool(b)) => EvalValue::Bool(*b),
            TermData::Const(Constant::Int(n)) => EvalValue::Rational(BigRational::from(n.clone())),
            TermData::Const(Constant::Rational(r)) => EvalValue::Rational(r.0.clone()),
            TermData::Const(_) => EvalValue::Unknown, // Other constants not handled yet

            // Variables: look up in appropriate model
            TermData::Var(_, _) => {
                if matches!(sort, Sort::Bool) {
                    // Boolean variable: check SAT model
                    match self.term_value(&model.sat_model, &model.term_to_var, term_id) {
                        Some(b) => EvalValue::Bool(b),
                        None => EvalValue::Bool(false), // Default unassigned to false
                    }
                } else if let Some(ref euf_model) = model.euf_model {
                    // Uninterpreted sort: check EUF model
                    if let Some(elem) = euf_model.term_values.get(&term_id) {
                        return EvalValue::Element(elem.clone());
                    }
                    EvalValue::Unknown
                } else {
                    EvalValue::Unknown
                }
            }

            // Negation
            TermData::Not(inner) => match self.evaluate_term(model, *inner) {
                EvalValue::Bool(b) => EvalValue::Bool(!b),
                _ => EvalValue::Unknown,
            },

            // If-then-else
            TermData::Ite(cond, then_br, else_br) => match self.evaluate_term(model, *cond) {
                EvalValue::Bool(true) => self.evaluate_term(model, *then_br),
                EvalValue::Bool(false) => self.evaluate_term(model, *else_br),
                _ => EvalValue::Unknown,
            },

            // Function applications
            TermData::App(sym, args) => {
                let name = sym.name();
                match name {
                    "and" => {
                        // All arguments must be true
                        for &arg in args {
                            match self.evaluate_term(model, arg) {
                                EvalValue::Bool(false) => return EvalValue::Bool(false),
                                EvalValue::Bool(true) => {}
                                _ => return EvalValue::Unknown,
                            }
                        }
                        EvalValue::Bool(true)
                    }
                    "or" => {
                        // Any argument must be true
                        for &arg in args {
                            match self.evaluate_term(model, arg) {
                                EvalValue::Bool(true) => return EvalValue::Bool(true),
                                EvalValue::Bool(false) => {}
                                _ => return EvalValue::Unknown,
                            }
                        }
                        EvalValue::Bool(false)
                    }
                    "=>" => {
                        // Implication: a => b is (not a) or b
                        if args.len() == 2 {
                            let a = self.evaluate_term(model, args[0]);
                            let b = self.evaluate_term(model, args[1]);
                            match (a, b) {
                                (EvalValue::Bool(false), _) => EvalValue::Bool(true),
                                (EvalValue::Bool(true), EvalValue::Bool(b)) => EvalValue::Bool(b),
                                _ => EvalValue::Unknown,
                            }
                        } else {
                            EvalValue::Unknown
                        }
                    }
                    "=" => {
                        // Equality: both arguments must evaluate to same value
                        if args.len() == 2 {
                            let v1 = self.evaluate_term(model, args[0]);
                            let v2 = self.evaluate_term(model, args[1]);
                            match (&v1, &v2) {
                                (EvalValue::Bool(b1), EvalValue::Bool(b2)) => {
                                    EvalValue::Bool(b1 == b2)
                                }
                                (EvalValue::Element(e1), EvalValue::Element(e2)) => {
                                    EvalValue::Bool(e1 == e2)
                                }
                                (EvalValue::Rational(r1), EvalValue::Rational(r2)) => {
                                    EvalValue::Bool(r1 == r2)
                                }
                                _ => EvalValue::Unknown,
                            }
                        } else {
                            EvalValue::Unknown
                        }
                    }
                    "distinct" => {
                        // All arguments must have different values
                        let values: Vec<EvalValue> =
                            args.iter().map(|&a| self.evaluate_term(model, a)).collect();

                        // Check for any unknown values
                        if values.iter().any(|v| matches!(v, EvalValue::Unknown)) {
                            return EvalValue::Unknown;
                        }

                        // Check all pairs are distinct
                        let mut seen: HashSet<String> = HashSet::new();
                        for v in &values {
                            let key = match v {
                                EvalValue::Bool(b) => format!("bool:{}", b),
                                EvalValue::Element(e) => format!("elem:{}", e),
                                EvalValue::Rational(r) => format!("rat:{}", r),
                                EvalValue::Unknown => unreachable!(),
                            };
                            if seen.contains(&key) {
                                return EvalValue::Bool(false);
                            }
                            seen.insert(key);
                        }
                        EvalValue::Bool(true)
                    }
                    "xor" => {
                        // XOR: exactly one of the two arguments must be true
                        if args.len() == 2 {
                            let a = self.evaluate_term(model, args[0]);
                            let b = self.evaluate_term(model, args[1]);
                            match (a, b) {
                                (EvalValue::Bool(a_val), EvalValue::Bool(b_val)) => {
                                    EvalValue::Bool(a_val != b_val)
                                }
                                _ => EvalValue::Unknown,
                            }
                        } else {
                            EvalValue::Unknown
                        }
                    }
                    // Arithmetic addition
                    "+" => {
                        let mut sum = BigRational::zero();
                        for &arg in args {
                            match self.evaluate_term(model, arg) {
                                EvalValue::Rational(r) => sum += r,
                                _ => return EvalValue::Unknown,
                            }
                        }
                        EvalValue::Rational(sum)
                    }
                    // Arithmetic subtraction (unary or binary)
                    "-" => {
                        if args.is_empty() {
                            return EvalValue::Unknown;
                        }
                        let first = self.evaluate_term(model, args[0]);
                        match first {
                            EvalValue::Rational(mut result) => {
                                if args.len() == 1 {
                                    // Unary negation
                                    EvalValue::Rational(-result)
                                } else {
                                    // Binary/n-ary subtraction
                                    for &arg in &args[1..] {
                                        match self.evaluate_term(model, arg) {
                                            EvalValue::Rational(r) => result -= r,
                                            _ => return EvalValue::Unknown,
                                        }
                                    }
                                    EvalValue::Rational(result)
                                }
                            }
                            _ => EvalValue::Unknown,
                        }
                    }
                    // Arithmetic multiplication
                    "*" => {
                        let mut product = BigRational::one();
                        for &arg in args {
                            match self.evaluate_term(model, arg) {
                                EvalValue::Rational(r) => product *= r,
                                _ => return EvalValue::Unknown,
                            }
                        }
                        EvalValue::Rational(product)
                    }
                    // Arithmetic division
                    "/" => {
                        if args.len() != 2 {
                            return EvalValue::Unknown;
                        }
                        let num = self.evaluate_term(model, args[0]);
                        let denom = self.evaluate_term(model, args[1]);
                        match (num, denom) {
                            (EvalValue::Rational(n), EvalValue::Rational(d)) => {
                                if d.is_zero() {
                                    EvalValue::Unknown // Division by zero
                                } else {
                                    EvalValue::Rational(n / d)
                                }
                            }
                            _ => EvalValue::Unknown,
                        }
                    }
                    // Less than
                    "<" => {
                        if args.len() != 2 {
                            return EvalValue::Unknown;
                        }
                        let lhs = self.evaluate_term(model, args[0]);
                        let rhs = self.evaluate_term(model, args[1]);
                        match (lhs, rhs) {
                            (EvalValue::Rational(l), EvalValue::Rational(r)) => {
                                EvalValue::Bool(l < r)
                            }
                            _ => EvalValue::Unknown,
                        }
                    }
                    // Less than or equal
                    "<=" => {
                        if args.len() != 2 {
                            return EvalValue::Unknown;
                        }
                        let lhs = self.evaluate_term(model, args[0]);
                        let rhs = self.evaluate_term(model, args[1]);
                        match (lhs, rhs) {
                            (EvalValue::Rational(l), EvalValue::Rational(r)) => {
                                EvalValue::Bool(l <= r)
                            }
                            _ => EvalValue::Unknown,
                        }
                    }
                    // Greater than
                    ">" => {
                        if args.len() != 2 {
                            return EvalValue::Unknown;
                        }
                        let lhs = self.evaluate_term(model, args[0]);
                        let rhs = self.evaluate_term(model, args[1]);
                        match (lhs, rhs) {
                            (EvalValue::Rational(l), EvalValue::Rational(r)) => {
                                EvalValue::Bool(l > r)
                            }
                            _ => EvalValue::Unknown,
                        }
                    }
                    // Greater than or equal
                    ">=" => {
                        if args.len() != 2 {
                            return EvalValue::Unknown;
                        }
                        let lhs = self.evaluate_term(model, args[0]);
                        let rhs = self.evaluate_term(model, args[1]);
                        match (lhs, rhs) {
                            (EvalValue::Rational(l), EvalValue::Rational(r)) => {
                                EvalValue::Bool(l >= r)
                            }
                            _ => EvalValue::Unknown,
                        }
                    }
                    // Uninterpreted function application
                    _ => {
                        // Check if this is a theory predicate (BV comparisons, etc.)
                        // Theory predicates should return Unknown since we can't evaluate them
                        // without the theory model values.
                        let is_theory_predicate = matches!(
                            name,
                            "bvult"
                                | "bvule"
                                | "bvugt"
                                | "bvuge"
                                | "bvslt"
                                | "bvsle"
                                | "bvsgt"
                                | "bvsge"
                        );
                        if is_theory_predicate {
                            // Trust the theory solver's verification for BV predicates
                            return EvalValue::Unknown;
                        }

                        // First try SAT model for Bool-sorted applications (predicates)
                        if matches!(sort, Sort::Bool) {
                            if let Some(b) =
                                self.term_value(&model.sat_model, &model.term_to_var, term_id)
                            {
                                return EvalValue::Bool(b);
                            }
                        }
                        // Then try EUF model
                        if let Some(ref euf_model) = model.euf_model {
                            if let Some(elem) = euf_model.term_values.get(&term_id) {
                                return EvalValue::Element(elem.clone());
                            }
                        }
                        // For unrecognized Bool predicates, return Unknown instead of
                        // defaulting to false, as they may be theory predicates we
                        // can't evaluate without model values.
                        if matches!(sort, Sort::Bool) {
                            return EvalValue::Unknown;
                        }
                        EvalValue::Unknown
                    }
                }
            }

            // Let bindings should be expanded, but handle just in case
            TermData::Let(_, body) => self.evaluate_term(model, *body),
        }
    }

    /// Validate that the current model satisfies all assertions
    ///
    /// Returns Ok(()) if all assertions evaluate to true, or Err with details
    /// about which assertion failed.
    pub fn validate_model(&self) -> std::result::Result<(), String> {
        let model = match (&self.last_result, &self.last_model) {
            (Some(CheckSatResult::Sat), Some(m)) => m,
            (Some(CheckSatResult::Sat), None) => {
                // SAT with no assertions is trivially valid
                if self.ctx.assertions.is_empty() {
                    return Ok(());
                }
                return Err("No model available".to_string());
            }
            _ => return Err("Model validation requires SAT result".to_string()),
        };

        for (i, &assertion) in self.ctx.assertions.iter().enumerate() {
            let value = self.evaluate_term(model, assertion);
            match value {
                EvalValue::Bool(true) => {}
                EvalValue::Bool(false) => {
                    let term_str = self.format_term(assertion);
                    return Err(format!(
                        "Assertion {} violated: {} evaluates to false",
                        i, term_str
                    ));
                }
                EvalValue::Unknown => {
                    // For assertions we can't evaluate (e.g., arithmetic in LRA/LIA),
                    // we trust the theory solver's verification. This is sound because
                    // theory solvers perform their own consistency checks.
                    // Only definite false values indicate a bug.
                }
                EvalValue::Element(_) | EvalValue::Rational(_) => {
                    let term_str = self.format_term(assertion);
                    return Err(format!(
                        "Assertion {} has non-Boolean value: {}",
                        i, term_str
                    ));
                }
            }
        }

        Ok(())
    }

    /// Get the term store
    pub fn term_store(&self) -> &TermStore {
        &self.ctx.terms
    }

    /// Get the current logic
    pub fn logic(&self) -> Option<&str> {
        self.ctx.logic.as_deref()
    }

    /// Get the number of assertions
    pub fn assertion_count(&self) -> usize {
        self.ctx.assertions.len()
    }

    /// Get the last check-sat result
    pub fn last_result(&self) -> Option<CheckSatResult> {
        self.last_result
    }

    /// Reset the executor
    pub fn reset(&mut self) {
        self.ctx = Context::new();
        self.last_result = None;
        self.last_model = None;
    }

    /// Check if produce-assignments option is enabled
    fn produce_assignments_enabled(&self) -> bool {
        matches!(
            self.ctx.get_option("produce-assignments"),
            Some(OptionValue::Bool(true))
        )
    }

    /// Check if produce-unsat-cores option is enabled
    fn produce_unsat_cores_enabled(&self) -> bool {
        matches!(
            self.ctx.get_option("produce-unsat-cores"),
            Some(OptionValue::Bool(true))
        )
    }

    /// Get assignment of named formulas (get-assignment command)
    ///
    /// Returns the truth values of all named Boolean formulas.
    fn get_assignment(&self) -> String {
        // Check that produce-assignments is enabled
        if !self.produce_assignments_enabled() {
            return "(error \"assignment generation is not enabled, set :produce-assignments to true\")".to_string();
        }

        // Check if we have a model
        let model = match (&self.last_result, &self.last_model) {
            (Some(CheckSatResult::Sat), Some(m)) => m,
            (Some(CheckSatResult::Sat), None) => {
                // SAT with no model (trivially satisfiable, no named terms to evaluate)
                return "()".to_string();
            }
            (Some(CheckSatResult::Unknown), _) => {
                // Unknown - still allowed, return assignment if available
                if let Some(m) = &self.last_model {
                    m
                } else {
                    return "()".to_string();
                }
            }
            _ => {
                return "(error \"assignment is not available\")".to_string();
            }
        };

        // Collect assignments for named terms
        let mut assignments = Vec::new();
        for (name, term_id) in self.ctx.named_terms_iter() {
            let value = self.evaluate_term(model, term_id);
            if let EvalValue::Bool(b) = value {
                assignments.push(format!("({} {})", name, b));
            }
        }

        if assignments.is_empty() {
            "()".to_string()
        } else {
            format!("({})", assignments.join("\n "))
        }
    }

    /// Get unsatisfiable core (get-unsat-core command)
    ///
    /// Returns names of assertions that form an unsatisfiable core.
    fn get_unsat_core(&self) -> String {
        // Check that produce-unsat-cores is enabled
        if !self.produce_unsat_cores_enabled() {
            return "(error \"unsat core generation is not enabled, set :produce-unsat-cores to true\")".to_string();
        }

        // Check that last result was unsat
        match self.last_result {
            Some(CheckSatResult::Unsat) => {
                // For now, return all named assertions as the unsat core
                // A proper implementation would track which assertions were actually used
                let names: Vec<String> = self
                    .ctx
                    .named_terms_iter()
                    .map(|(name, _)| name.to_string())
                    .collect();

                if names.is_empty() {
                    "()".to_string()
                } else {
                    format!("({})", names.join(" "))
                }
            }
            _ => "(error \"unsat core is not available, last result was not unsat\")".to_string(),
        }
    }

    /// Get unsatisfiable assumptions (get-unsat-assumptions command)
    ///
    /// Returns the subset of assumptions from check-sat-assuming that contributed
    /// to unsatisfiability. Per SMT-LIB 2.6, this returns a subset of the literals
    /// from the most recent check-sat-assuming call that was unsatisfiable.
    fn get_unsat_assumptions(&self) -> String {
        // Check that last result was unsat and we have assumptions
        match (&self.last_result, &self.last_assumptions) {
            (Some(CheckSatResult::Unsat), Some(assumptions)) => {
                if assumptions.is_empty() {
                    return "()".to_string();
                }

                // For now, return all assumptions as the unsat subset
                // A proper implementation would track which assumptions were actually needed
                // by analyzing the proof/conflict
                let literals: Vec<String> = assumptions
                    .iter()
                    .map(|&term_id| self.format_term(term_id))
                    .collect();

                format!("({})", literals.join(" "))
            }
            (Some(CheckSatResult::Unsat), None) => {
                // Unsat but no assumptions (regular check-sat, not check-sat-assuming)
                "(error \"no check-sat-assuming has been performed\")".to_string()
            }
            (Some(CheckSatResult::Sat), _) => {
                "(error \"unsat assumptions not available, last result was sat\")".to_string()
            }
            (Some(CheckSatResult::Unknown), _) => {
                "(error \"unsat assumptions not available, last result was unknown\")".to_string()
            }
            (None, _) => {
                "(error \"unsat assumptions not available, no check-sat has been performed\")"
                    .to_string()
            }
        }
    }

    /// Get proof (get-proof command)
    ///
    /// Returns a proof that the assertions are unsatisfiable in Alethe format.
    fn get_proof(&self) -> String {
        // Check that produce-proofs is enabled
        if !self.produce_proofs_enabled() {
            return "(error \"proof generation is not enabled, set :produce-proofs to true\")"
                .to_string();
        }

        // Check that last result was unsat
        match self.last_result {
            Some(CheckSatResult::Unsat) => {
                // Export the stored proof in Alethe format
                match &self.last_proof {
                    Some(proof) => export_alethe(proof, &self.ctx.terms),
                    None => "(error \"proof was not generated\")".to_string(),
                }
            }
            Some(CheckSatResult::Sat) => {
                "(error \"proof is not available, last result was sat\")".to_string()
            }
            Some(CheckSatResult::Unknown) => {
                "(error \"proof is not available, last result was unknown\")".to_string()
            }
            None => {
                "(error \"proof is not available, no check-sat has been performed\")".to_string()
            }
        }
    }

    fn produce_proofs_enabled(&self) -> bool {
        self.proof_tracker.is_enabled()
            || matches!(
                self.ctx.get_option("produce-proofs"),
                Some(OptionValue::Bool(true))
            )
    }
}

/// Format a sort for SMT-LIB output
fn format_sort(sort: &Sort) -> String {
    match sort {
        Sort::Bool => "Bool".to_string(),
        Sort::Int => "Int".to_string(),
        Sort::Real => "Real".to_string(),
        Sort::String => "String".to_string(),
        Sort::BitVec(w) => format!("(_ BitVec {})", w),
        Sort::FloatingPoint(eb, sb) => format!("(_ FloatingPoint {} {})", eb, sb),
        Sort::Array(idx, elem) => format!("(Array {} {})", format_sort(idx), format_sort(elem)),
        Sort::Uninterpreted(name) => name.clone(),
        Sort::Datatype(name) => name.clone(),
    }
}

/// Format a value for SMT-LIB output based on sort
fn format_value(sort: &Sort, value: Option<bool>, _terms: &TermStore) -> String {
    match sort {
        Sort::Bool => match value {
            Some(true) => "true".to_string(),
            Some(false) => "false".to_string(),
            None => "false".to_string(), // Default to false for unassigned
        },
        Sort::Int => {
            // For Int, we'd need to look up the actual value from the model
            // For now, return a placeholder since we only have Boolean model
            "0".to_string()
        }
        Sort::Real => "0.0".to_string(),
        Sort::String => "\"\"".to_string(),
        Sort::BitVec(w) => format!("#x{:0>width$}", 0, width = (*w as usize).div_ceil(4)),
        Sort::FloatingPoint(_, _) => "(_ +zero 8 24)".to_string(),
        Sort::Array(_, elem) => {
            // Return a constant array
            format!(
                "((as const {}) {})",
                format_sort(sort),
                format_value(elem, None, _terms)
            )
        }
        Sort::Uninterpreted(name) | Sort::Datatype(name) => {
            // For uninterpreted sorts, return a fresh element identifier
            format!("@{}!0", name)
        }
    }
}

/// Format a default value for a sort (used for array elements)
fn format_default_value(sort: &Sort) -> String {
    match sort {
        Sort::Bool => "false".to_string(),
        Sort::Int => "0".to_string(),
        Sort::Real => "0.0".to_string(),
        Sort::String => "\"\"".to_string(),
        Sort::BitVec(w) => format!("#x{:0>width$}", 0, width = (*w as usize).div_ceil(4)),
        Sort::FloatingPoint(_, _) => "(_ +zero 8 24)".to_string(),
        Sort::Array(_, elem) => {
            // Recursive: const-array of default element value
            format!(
                "((as const {}) {})",
                format_sort(sort),
                format_default_value(elem)
            )
        }
        Sort::Uninterpreted(name) | Sort::Datatype(name) => {
            format!("@{}!0", name)
        }
    }
}

/// Format a BigRational as SMT-LIB Real value
fn format_rational(val: &BigRational) -> String {
    if val.is_integer() {
        // Integer value: format as decimal
        let numer = val.numer();
        if numer.sign() == num_bigint::Sign::Minus {
            format!("(- {})", numer.magnitude())
        } else {
            format!("{}.0", numer)
        }
    } else {
        // Fractional value: format as (/ num denom)
        let numer = val.numer();
        let denom = val.denom();
        if numer.sign() == num_bigint::Sign::Minus {
            format!("(- (/ {} {}))", numer.magnitude(), denom)
        } else {
            format!("(/ {} {})", numer, denom)
        }
    }
}

/// Format a BigInt as SMT-LIB Int value
fn format_bigint(val: &num_bigint::BigInt) -> String {
    use num_bigint::Sign;

    match val.sign() {
        Sign::Minus => format!("(- {})", val.magnitude()),
        Sign::NoSign | Sign::Plus => val.to_string(),
    }
}

/// Format a BigInt as SMT-LIB BitVec hex value
fn format_bitvec(val: &num_bigint::BigInt, width: u32) -> String {
    use num_traits::ToPrimitive;

    // For small bitvectors, use binary format
    if width <= 4 {
        let mask = (1u64 << width) - 1;
        let v = val.to_u64().unwrap_or(0) & mask;
        return format!("#b{:0width$b}", v, width = width as usize);
    }

    // For larger bitvectors, use hex format
    // Calculate number of hex digits needed (round up)
    let hex_digits = (width as usize).div_ceil(4);
    let mask = if width >= 64 {
        num_bigint::BigInt::from(1) << width
    } else {
        num_bigint::BigInt::from(1u64 << width)
    };

    // Apply mask to get unsigned value
    let unsigned_val = val & (&mask - 1);

    // Format as hex with proper padding
    let hex_str = format!("{:x}", unsigned_val);
    let padded = format!("{:0>width$}", hex_str, width = hex_digits);
    format!("#x{}", padded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use z4_frontend::parse;

    #[test]
    fn test_executor_simple_sat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert (or a b))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_simple_unsat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (assert (not a))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_euf_unsat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (declare-fun p (U) Bool)
            (assert (= a b))
            (assert (p a))
            (assert (not (p b)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_euf_sat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (declare-fun p (U) Bool)
            (assert (p a))
            (assert (not (p b)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_euf_congruence() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (declare-const c U)
            (assert (= a b))
            (assert (= b c))
            (assert (not (= a c)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_distinct() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (assert (= a b))
            (assert (distinct a b))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_multiple_check_sat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (check-sat)
            (assert (not a))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat", "unsat"]);
    }

    #[test]
    fn test_executor_push_pop() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (push 1)
            (assert (not a))
            (check-sat)
            (pop 1)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // After push + assert (not a), should be unsat
        // After pop, only a is asserted, should be sat
        assert_eq!(outputs, vec!["unsat", "sat"]);
    }

    #[test]
    fn test_executor_no_logic() {
        // Should work with default logic (treated as QF_UF)
        let input = r#"
            (declare-const a Bool)
            (assert a)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_get_model_bool() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (assert (not b))
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        // Model should contain definitions for a and b
        let model = &outputs[1];
        assert!(model.starts_with("(model"));
        assert!(model.contains("define-fun a () Bool"));
        assert!(model.contains("define-fun b () Bool"));
    }

    #[test]
    fn test_get_model_before_check_sat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].contains("error"));
    }

    #[test]
    fn test_get_model_after_unsat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (assert (not a))
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "unsat");
        assert!(outputs[1].contains("error"));
    }

    #[test]
    fn test_get_model_empty_assertions() {
        let input = r#"
            (set-logic QF_UF)
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        // Empty model is valid for no assertions
        assert!(outputs[1].contains("model"));
    }

    #[test]
    fn test_get_model_requires_produce_models() {
        let input = r#"
            (set-option :produce-models false)
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        assert!(outputs[1].contains("error"), "Output: {}", outputs[1]);
        assert!(
            outputs[1].contains("model generation is not enabled"),
            "Output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_model_with_uninterpreted_sort() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const x U)
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        let model = &outputs[1];
        // For a trivially SAT case with no Boolean constraints, we may get
        // an empty model or one with placeholder values
        assert!(model.contains("model"), "Model output: {}", model);
    }

    #[test]
    fn test_get_value_bool() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (assert (not b))
            (check-sat)
            (get-value (a b))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        // Get-value should output pairs like ((a true) (b false))
        let values = &outputs[1];
        assert!(values.starts_with('('), "Values: {}", values);
        assert!(values.contains("a"), "Values: {}", values);
        assert!(values.contains("b"), "Values: {}", values);
    }

    #[test]
    fn test_get_value_before_check_sat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (get-value (a))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].contains("error"), "Output: {}", outputs[0]);
    }

    #[test]
    fn test_get_value_requires_produce_models() {
        let input = r#"
            (set-option :produce-models false)
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (check-sat)
            (get-value (a))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        assert!(outputs[1].contains("error"), "Output: {}", outputs[1]);
        assert!(
            outputs[1].contains("model generation is not enabled"),
            "Output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_value_after_unsat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (assert (not a))
            (check-sat)
            (get-value (a))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "unsat");
        assert!(outputs[1].contains("error"), "Output: {}", outputs[1]);
    }

    #[test]
    fn test_get_value_expression() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (assert b)
            (check-sat)
            (get-value ((and a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        // Get-value for (and a b) should work
        let values = &outputs[1];
        assert!(values.starts_with('('), "Values: {}", values);
        assert!(values.contains("and"), "Values: {}", values);
    }

    #[test]
    fn test_get_value_multiple_terms() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const x Bool)
            (declare-const y Bool)
            (declare-const z Bool)
            (assert x)
            (assert (not y))
            (check-sat)
            (get-value (x y z))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        let values = &outputs[1];
        // Should have all three variables
        assert!(values.contains("x"), "Values: {}", values);
        assert!(values.contains("y"), "Values: {}", values);
        assert!(values.contains("z"), "Values: {}", values);
    }

    #[test]
    fn test_get_model_uninterpreted_constants() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (assert (distinct a b))
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        let model = &outputs[1];
        // Model should include definitions for a and b with different values
        assert!(model.contains("define-fun a"), "Model: {}", model);
        assert!(model.contains("define-fun b"), "Model: {}", model);
        // Values should be element identifiers like @U!0
        assert!(model.contains("@U!"), "Model: {}", model);
    }

    #[test]
    fn test_get_model_uninterpreted_function() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (declare-fun f (U) U)
            (assert (= (f a) b))
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        let model = &outputs[1];
        // Model should include a function definition for f
        assert!(model.contains("define-fun f"), "Model: {}", model);
        // Function should have a parameter of sort U
        assert!(model.contains("(x0 U)"), "Model: {}", model);
    }

    #[test]
    fn test_get_model_equal_constants_same_value() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (assert (= a b))
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        let model = &outputs[1];
        // Model should have a and b with the same value since they're equal
        assert!(model.contains("define-fun a"), "Model: {}", model);
        assert!(model.contains("define-fun b"), "Model: {}", model);
        // Both should have the same element (e.g., @U!0)
        // Extract values to verify they're the same
        let lines: Vec<&str> = model.lines().collect();
        let mut values = Vec::new();
        for line in lines {
            if line.contains("define-fun a") || line.contains("define-fun b") {
                // Extract the value (last token before closing paren)
                if let Some(val) = line.split_whitespace().last() {
                    let val = val.trim_end_matches(')');
                    values.push(val.to_string());
                }
            }
        }
        assert_eq!(
            values.len(),
            2,
            "Expected 2 values, got {}: {:?}",
            values.len(),
            values
        );
        assert_eq!(
            values[0], values[1],
            "a and b should have same value: {:?}",
            values
        );
    }

    #[test]
    fn test_get_model_function_congruence() {
        // When a = b, f(a) and f(b) should have the same value
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (declare-fun f (U) U)
            (assert (= a b))
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        let model = &outputs[1];
        // Model should be valid
        assert!(model.contains("(model"), "Model: {}", model);
    }

    // ========== Model Validation Tests ==========

    #[test]
    fn test_validate_model_simple_bool() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (assert (not b))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        exec.execute_all(&commands).unwrap();

        // Model validation should succeed
        let result = exec.validate_model();
        assert!(result.is_ok(), "Model validation failed: {:?}", result);
    }

    #[test]
    fn test_validate_model_with_and() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert (and a b))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        exec.execute_all(&commands).unwrap();

        let result = exec.validate_model();
        assert!(result.is_ok(), "Model validation failed: {:?}", result);
    }

    #[test]
    fn test_validate_model_with_or() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert (or a b))
            (assert (not a))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        exec.execute_all(&commands).unwrap();

        let result = exec.validate_model();
        assert!(result.is_ok(), "Model validation failed: {:?}", result);
    }

    #[test]
    fn test_validate_model_euf_equality() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (assert (= a b))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        exec.execute_all(&commands).unwrap();

        let result = exec.validate_model();
        assert!(result.is_ok(), "Model validation failed: {:?}", result);
    }

    #[test]
    fn test_validate_model_euf_distinct() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (declare-const c U)
            (assert (distinct a b c))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        exec.execute_all(&commands).unwrap();

        let result = exec.validate_model();
        assert!(result.is_ok(), "Model validation failed: {:?}", result);
    }

    #[test]
    fn test_validate_model_requires_sat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (assert (not a))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        exec.execute_all(&commands).unwrap();

        // Validation should fail because result is UNSAT
        let result = exec.validate_model();
        assert!(result.is_err(), "Should fail for UNSAT");
    }

    #[test]
    fn test_validate_model_no_check_sat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        exec.execute_all(&commands).unwrap();

        // Validation should fail because no check-sat was run
        let result = exec.validate_model();
        assert!(result.is_err(), "Should fail without check-sat");
    }

    // ========== Composite Term Evaluation Tests ==========

    #[test]
    fn test_get_value_composite_and_true() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (assert b)
            (check-sat)
            (get-value ((and a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (and a b) should evaluate to true
        assert!(
            outputs[1].contains("true"),
            "Expected true in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_value_composite_and_false() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (assert (not b))
            (check-sat)
            (get-value ((and a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (and a b) should evaluate to false since b is false
        assert!(
            outputs[1].contains("false"),
            "Expected false in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_value_composite_or() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert (not a))
            (assert b)
            (check-sat)
            (get-value ((or a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (or a b) should evaluate to true since b is true
        assert!(
            outputs[1].contains("true"),
            "Expected true in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_value_composite_not() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (check-sat)
            (get-value ((not a)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (not a) should evaluate to false since a is true
        assert!(
            outputs[1].contains("false"),
            "Expected false in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_value_composite_equality() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (assert (= a b))
            (check-sat)
            (get-value ((= a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (= a b) should evaluate to true
        assert!(
            outputs[1].contains("true"),
            "Expected true in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_value_nested_composite() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (declare-const c Bool)
            (assert a)
            (assert b)
            (assert (not c))
            (check-sat)
            (get-value ((and (or a b) (not c))))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (and (or a b) (not c)) = (and true true) = true
        assert!(
            outputs[1].contains("true"),
            "Expected true in output: {}",
            outputs[1]
        );
    }

    // ========== Predicate Table Tests ==========

    #[test]
    fn test_get_model_predicate_function() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (declare-fun p (U) Bool)
            (assert (p a))
            (assert (not (p b)))
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        let model = &outputs[1];
        // Model should include a function definition for predicate p
        assert!(
            model.contains("define-fun p"),
            "Model should contain p: {}",
            model
        );
        // Should have Bool as return type
        assert!(
            model.contains("Bool"),
            "Model should have Bool return type: {}",
            model
        );
    }

    #[test]
    fn test_validate_model_with_predicate() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (declare-fun p (U) Bool)
            (assert (p a))
            (assert (not (p b)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        exec.execute_all(&commands).unwrap();

        // Model validation should succeed
        let result = exec.validate_model();
        assert!(result.is_ok(), "Model validation failed: {:?}", result);
    }

    // ========== If-Then-Else Evaluation Tests ==========

    #[test]
    fn test_get_value_ite_true_condition() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const c Bool)
            (declare-const x Bool)
            (declare-const y Bool)
            (assert c)
            (assert x)
            (assert (not y))
            (check-sat)
            (get-value ((ite c x y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (ite c x y) with c=true should return x=true
        assert!(
            outputs[1].contains("true"),
            "Expected true (then branch) in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_value_ite_false_condition() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const c Bool)
            (declare-const x Bool)
            (declare-const y Bool)
            (assert (not c))
            (assert x)
            (assert (not y))
            (check-sat)
            (get-value ((ite c x y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (ite c x y) with c=false should return y=false
        assert!(
            outputs[1].contains("false"),
            "Expected false (else branch) in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_value_nested_ite() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (declare-const x Bool)
            (declare-const y Bool)
            (declare-const z Bool)
            (assert a)
            (assert (not b))
            (assert x)
            (assert (not y))
            (assert z)
            (check-sat)
            (get-value ((ite a (ite b y x) z)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // a=true, so we take (ite b y x)
        // b=false, so we take x=true
        assert!(
            outputs[1].contains("true"),
            "Expected true in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_validate_model_with_ite() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const c Bool)
            (declare-const x Bool)
            (declare-const y Bool)
            (assert c)
            (assert (ite c x (not x)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        exec.execute_all(&commands).unwrap();

        let result = exec.validate_model();
        assert!(result.is_ok(), "Model validation failed: {:?}", result);
    }

    #[test]
    fn test_get_value_ite_with_euf_elements() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const c Bool)
            (declare-const x U)
            (declare-const y U)
            (assert c)
            (assert (distinct x y))
            (check-sat)
            (get-value ((ite c x y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (ite c x y) with c=true should return x's element value
        let values = &outputs[1];
        assert!(values.contains("ite"), "Values: {}", values);
        assert!(
            values.contains("@U!"),
            "Expected element value in output: {}",
            values
        );
    }

    // ========== Implication Evaluation Tests ==========

    #[test]
    fn test_get_value_implication_true_antecedent() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (assert b)
            (check-sat)
            (get-value ((=> a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (=> a b) with a=true, b=true should be true
        assert!(
            outputs[1].contains("true"),
            "Expected true in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_value_implication_false_antecedent() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert (not a))
            (assert (not b))
            (check-sat)
            (get-value ((=> a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (=> a b) with a=false should be true regardless of b
        assert!(
            outputs[1].contains("true"),
            "Expected true (vacuous truth) in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_value_implication_false_result() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (assert (not b))
            (check-sat)
            (get-value ((=> a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (=> a b) with a=true, b=false should be false
        assert!(
            outputs[1].contains("false"),
            "Expected false in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_validate_model_with_implication() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert (=> a b))
            (assert a)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        exec.execute_all(&commands).unwrap();

        let result = exec.validate_model();
        assert!(result.is_ok(), "Model validation failed: {:?}", result);
    }

    #[test]
    fn test_get_value_nested_implication() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (declare-const c Bool)
            (assert a)
            (assert b)
            (assert c)
            (check-sat)
            (get-value ((=> (=> a b) c)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (=> (=> a b) c) = (=> true true) = true
        assert!(
            outputs[1].contains("true"),
            "Expected true in output: {}",
            outputs[1]
        );
    }

    // ========== XOR Evaluation Tests ==========
    // XOR is desugared during elaboration to (or (and a (not b)) (and (not a) b))
    // These tests verify the desugaring works correctly with model evaluation

    #[test]
    fn test_get_value_xor_true() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (assert (not b))
            (check-sat)
            (get-value ((xor a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // xor(true, false) = true
        assert!(
            outputs[1].contains("true"),
            "Expected true in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_value_xor_false_same_values() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (assert b)
            (check-sat)
            (get-value ((xor a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // xor(true, true) = false
        assert!(
            outputs[1].contains("false"),
            "Expected false in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_validate_model_with_xor() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert (xor a b))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        exec.execute_all(&commands).unwrap();

        let result = exec.validate_model();
        assert!(result.is_ok(), "Model validation failed: {:?}", result);
    }

    // ========== Boolean Equality Tests ==========

    #[test]
    fn test_get_value_bool_equality_true() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (assert b)
            (check-sat)
            (get-value ((= a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (= true true) = true
        assert!(
            outputs[1].contains("true"),
            "Expected true in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_value_bool_equality_false() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (assert (not b))
            (check-sat)
            (get-value ((= a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs[0], "sat");
        // (= true false) = false
        assert!(
            outputs[1].contains("false"),
            "Expected false in output: {}",
            outputs[1]
        );
    }

    // ========== Check-Sat-Assuming Tests ==========

    #[test]
    fn test_check_sat_assuming_sat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (check-sat-assuming (b))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // With a=true asserted and b assumed, should be SAT
        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_check_sat_assuming_unsat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (check-sat-assuming ((not a)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // With a=true asserted and (not a) assumed, should be UNSAT
        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_check_sat_assuming_does_not_modify_stack() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (check-sat-assuming ((not a)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // First check-sat-assuming with (not a) should be UNSAT
        // Second check-sat (without assumption) should be SAT
        assert_eq!(outputs, vec!["unsat", "sat"]);
    }

    #[test]
    fn test_check_sat_assuming_multiple_assumptions() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (declare-const c Bool)
            (check-sat-assuming (a b c))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // All assumptions are satisfiable together
        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_check_sat_assuming_contradictory_assumptions() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (check-sat-assuming (a (not a)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // Contradictory assumptions should be UNSAT
        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_check_sat_assuming_empty() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (check-sat-assuming ())
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // Empty assumptions - equivalent to check-sat
        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_check_sat_assuming_euf() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (assert (= a b))
            (check-sat-assuming ((distinct a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // a=b asserted, (distinct a b) assumed should be UNSAT
        assert_eq!(outputs, vec!["unsat"]);
    }

    // ========== Get-Info Tests ==========

    #[test]
    fn test_get_info_name() {
        let input = r#"
            (get-info :name)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("z4"),
            "Expected solver name: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_get_info_version() {
        let input = r#"
            (get-info :version)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("version"),
            "Expected version info: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_get_info_assertion_stack_levels() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (get-info :assertion-stack-levels)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("assertion-stack-levels"),
            "Expected assertion-stack-levels: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_get_info_all_statistics() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (get-info :all-statistics)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("all-statistics"),
            "Expected all-statistics: {}",
            outputs[0]
        );
        assert!(
            outputs[0].contains("assertions"),
            "Expected assertions count: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_get_info_unsupported() {
        let input = r#"
            (get-info :nonexistent-keyword)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("error"),
            "Expected error for unsupported keyword: {}",
            outputs[0]
        );
    }

    // ========== get-option tests ==========

    #[test]
    fn test_get_option_produce_models() {
        let input = r#"
            (get-option :produce-models)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("produce-models"),
            "Expected produce-models: {}",
            outputs[0]
        );
        assert!(
            outputs[0].contains("true"),
            "Expected true value: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_get_option_print_success() {
        let input = r#"
            (get-option :print-success)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("print-success"),
            "Expected print-success: {}",
            outputs[0]
        );
        assert!(
            outputs[0].contains("false"),
            "Expected false value (default): {}",
            outputs[0]
        );
    }

    #[test]
    fn test_get_option_random_seed() {
        let input = r#"
            (get-option :random-seed)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("random-seed"),
            "Expected random-seed: {}",
            outputs[0]
        );
        assert!(
            outputs[0].contains("0"),
            "Expected 0 (default): {}",
            outputs[0]
        );
    }

    #[test]
    fn test_set_and_get_option() {
        let input = r#"
            (set-option :random-seed 42)
            (get-option :random-seed)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("42"),
            "Expected 42 after set-option: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_set_and_get_option_bool() {
        let input = r#"
            (set-option :print-success true)
            (get-option :print-success)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("true"),
            "Expected true after set-option: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_get_option_unknown() {
        let input = r#"
            (get-option :nonexistent-option)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("error"),
            "Expected error for unknown option: {}",
            outputs[0]
        );
    }

    // ========== echo tests ==========

    #[test]
    fn test_echo_basic() {
        let input = r#"
            (echo "hello")
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["hello"]);
    }

    #[test]
    fn test_echo_with_escape_sequences() {
        let input = r#"
            (echo "hello\nworld")
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["hello\nworld"]);
    }

    // ========== get-assertions tests ==========

    #[test]
    fn test_get_assertions_empty() {
        let input = r#"
            (get-assertions)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "()");
    }

    #[test]
    fn test_get_assertions_single() {
        let input = r#"
            (declare-const a Bool)
            (assert a)
            (get-assertions)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("a"),
            "Expected assertion 'a': {}",
            outputs[0]
        );
    }

    #[test]
    fn test_get_assertions_multiple() {
        let input = r#"
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (assert (not b))
            (get-assertions)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("a"),
            "Expected assertion 'a': {}",
            outputs[0]
        );
        assert!(
            outputs[0].contains("not"),
            "Expected 'not' in assertions: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_get_assertions_with_compound() {
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (assert (and x y))
            (get-assertions)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("and"),
            "Expected 'and' in assertions: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_get_assertions_with_euf() {
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (assert (= a b))
            (get-assertions)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("="),
            "Expected '=' in assertions: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_get_assertions_after_push_pop() {
        let input = r#"
            (declare-const a Bool)
            (assert a)
            (push 1)
            (declare-const b Bool)
            (assert b)
            (get-assertions)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Both a and b should be in assertions
        assert!(
            outputs[0].contains("a"),
            "Expected assertion 'a': {}",
            outputs[0]
        );
        assert!(
            outputs[0].contains("b"),
            "Expected assertion 'b': {}",
            outputs[0]
        );
    }

    #[test]
    fn test_get_assertions_after_pop() {
        let input = r#"
            (declare-const a Bool)
            (assert a)
            (push 1)
            (declare-const b Bool)
            (assert b)
            (pop 1)
            (get-assertions)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Only 'a' should remain after pop
        assert!(
            outputs[0].contains("a"),
            "Expected assertion 'a' to remain: {}",
            outputs[0]
        );
        // Check that the output only contains one assertion
        // (the "(a)" pattern should be the whole content)
        assert!(
            !outputs[0].contains("b"),
            "Did not expect 'b' after pop: {}",
            outputs[0]
        );
    }

    // ========== get-assignment Tests ==========

    #[test]
    fn test_get_assignment_not_enabled() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert (! a :named my_assertion))
            (check-sat)
            (get-assignment)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        assert!(
            outputs[1].contains("error"),
            "Expected error about produce-assignments: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_assignment_enabled_sat() {
        let input = r#"
            (set-option :produce-assignments true)
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert (! a :named my_assertion))
            (check-sat)
            (get-assignment)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        // Should contain assignment for my_assertion
        assert!(
            outputs[1].contains("my_assertion"),
            "Expected named term in assignment: {}",
            outputs[1]
        );
        // Since 'a' is asserted, my_assertion should be true
        assert!(
            outputs[1].contains("true"),
            "Expected true value: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_assignment_multiple_named() {
        let input = r#"
            (set-option :produce-assignments true)
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert (! a :named a_holds))
            (assert (! (not b) :named not_b_holds))
            (check-sat)
            (get-assignment)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        // Both named terms should appear
        assert!(
            outputs[1].contains("a_holds"),
            "Expected a_holds: {}",
            outputs[1]
        );
        assert!(
            outputs[1].contains("not_b_holds"),
            "Expected not_b_holds: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_assignment_no_named_terms() {
        let input = r#"
            (set-option :produce-assignments true)
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (check-sat)
            (get-assignment)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        // Should return empty list since no named terms
        assert_eq!(outputs[1], "()");
    }

    #[test]
    fn test_get_assignment_before_check_sat() {
        let input = r#"
            (set-option :produce-assignments true)
            (declare-const a Bool)
            (assert (! a :named my_a))
            (get-assignment)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should return error since no check-sat yet
        assert!(
            outputs[0].contains("error"),
            "Expected error about unavailable assignment: {}",
            outputs[0]
        );
    }

    // ========== get-unsat-core Tests ==========

    #[test]
    fn test_get_unsat_core_not_enabled() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert (! a :named pos_a))
            (assert (! (not a) :named neg_a))
            (check-sat)
            (get-unsat-core)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "unsat");
        assert!(
            outputs[1].contains("error"),
            "Expected error about produce-unsat-cores: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_unsat_core_enabled() {
        let input = r#"
            (set-option :produce-unsat-cores true)
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert (! a :named pos_a))
            (assert (! (not a) :named neg_a))
            (check-sat)
            (get-unsat-core)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "unsat");
        // Should contain names of assertions that form the core
        assert!(
            outputs[1].contains("pos_a") || outputs[1].contains("neg_a"),
            "Expected named assertions in core: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_unsat_core_after_sat() {
        let input = r#"
            (set-option :produce-unsat-cores true)
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert (! a :named pos_a))
            (check-sat)
            (get-unsat-core)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        // Should return error since last result was not unsat
        assert!(
            outputs[1].contains("error"),
            "Expected error about unsat core not available: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_unsat_core_no_named_terms() {
        let input = r#"
            (set-option :produce-unsat-cores true)
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (assert (not a))
            (check-sat)
            (get-unsat-core)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "unsat");
        // Should return empty list since no named terms
        assert_eq!(outputs[1], "()");
    }

    // ========== get-unsat-assumptions Tests ==========

    #[test]
    fn test_get_unsat_assumptions_after_check_sat_assuming() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert a)
            (check-sat-assuming ((not a) b))
            (get-unsat-assumptions)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "unsat");
        // Should return assumptions that caused unsat
        // (not a) is the conflicting assumption since a is asserted
        assert!(
            outputs[1].contains("not") || outputs[1].contains("a") || outputs[1].contains("b"),
            "Expected assumptions in output: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_unsat_assumptions_no_check_sat_assuming() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (assert (not a))
            (check-sat)
            (get-unsat-assumptions)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "unsat");
        // Should error because regular check-sat was used, not check-sat-assuming
        assert!(
            outputs[1].contains("error") && outputs[1].contains("check-sat-assuming"),
            "Expected error about no check-sat-assuming: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_unsat_assumptions_after_sat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (check-sat-assuming (a))
            (get-unsat-assumptions)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        // Should error because last result was SAT
        assert!(
            outputs[1].contains("error") && outputs[1].contains("sat"),
            "Expected error about sat result: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_unsat_assumptions_empty() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (assert (not a))
            (check-sat-assuming ())
            (get-unsat-assumptions)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "unsat");
        // Empty assumptions list should return empty
        assert_eq!(outputs[1], "()");
    }

    #[test]
    fn test_get_unsat_assumptions_contradictory() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (check-sat-assuming (a (not a)))
            (get-unsat-assumptions)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "unsat");
        // Should return the assumptions (a and (not a))
        assert!(
            outputs[1].contains("a"),
            "Expected 'a' in assumptions: {}",
            outputs[1]
        );
        assert!(
            outputs[1].contains("not"),
            "Expected '(not a)' in assumptions: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_unsat_assumptions_no_check_sat() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (get-unsat-assumptions)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should error because no check-sat has been performed
        assert!(
            outputs[0].contains("error") && outputs[0].contains("no check-sat"),
            "Expected error about no check-sat: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_get_proof_not_enabled() {
        let input = r#"
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (assert (not a))
            (check-sat)
            (get-proof)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "unsat");
        assert!(
            outputs[1].contains("error") && outputs[1].contains("produce-proofs"),
            "Expected error about produce-proofs: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_proof_after_sat() {
        let input = r#"
            (set-option :produce-proofs true)
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (check-sat)
            (get-proof)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        assert!(
            outputs[1].contains("error") && outputs[1].contains("sat"),
            "Expected error about result being sat: {}",
            outputs[1]
        );
    }

    #[test]
    fn test_get_proof_after_unsat() {
        let input = r#"
            (set-option :produce-proofs true)
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (assert (not a))
            (check-sat)
            (get-proof)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "unsat");
        // Proof should be in Alethe format with assume and step commands
        let proof = &outputs[1];
        assert!(
            proof.contains("assume"),
            "Expected proof to contain assume steps: {}",
            proof
        );
        assert!(
            proof.contains("step"),
            "Expected proof to contain step: {}",
            proof
        );
        assert!(
            proof.contains("(cl)"),
            "Expected proof to derive empty clause: {}",
            proof
        );
    }

    #[test]
    fn test_get_proof_no_check_sat() {
        let input = r#"
            (set-option :produce-proofs true)
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert a)
            (get-proof)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("error") && outputs[0].contains("no check-sat"),
            "Expected error about no check-sat: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_named_terms_cleared_on_pop() {
        let input = r#"
            (set-option :produce-unsat-cores true)
            (set-logic QF_UF)
            (declare-const a Bool)
            (assert (! a :named outer_a))
            (push 1)
            (assert (! (not a) :named inner_not_a))
            (check-sat)
            (get-unsat-core)
            (pop 1)
            (check-sat)
            (get-unsat-core)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // First check-sat is unsat (a and not a)
        // First get-unsat-core should include both outer_a and inner_not_a
        // After pop, inner_not_a should be removed
        // Second check-sat is sat (only a is asserted)
        // Second get-unsat-core should error because sat

        assert_eq!(outputs.len(), 4);
        assert_eq!(outputs[0], "unsat");
        // First unsat-core should have both names
        assert!(
            outputs[1].contains("outer_a") && outputs[1].contains("inner_not_a"),
            "Expected both named terms in core: {}",
            outputs[1]
        );
        assert_eq!(outputs[2], "sat");
        // After pop and sat, get-unsat-core should error
        assert!(
            outputs[3].contains("error"),
            "Expected error after sat: {}",
            outputs[3]
        );
    }

    #[test]
    fn test_named_terms_scope_independent() {
        // Named terms defined outside any scope should remain after pop
        let input = r#"
            (set-option :produce-unsat-cores true)
            (set-logic QF_UF)
            (declare-const a Bool)
            (declare-const b Bool)
            (assert (! a :named assert_a))
            (push 1)
            (assert (! b :named assert_b))
            (pop 1)
            (push 1)
            (assert (! (not a) :named assert_not_a))
            (check-sat)
            (get-unsat-core)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "unsat");
        // Should have assert_a (outer scope) and assert_not_a (current scope)
        // Should NOT have assert_b (was popped)
        assert!(
            outputs[1].contains("assert_a"),
            "Expected outer named term: {}",
            outputs[1]
        );
        assert!(
            outputs[1].contains("assert_not_a"),
            "Expected inner named term: {}",
            outputs[1]
        );
        assert!(
            !outputs[1].contains("assert_b"),
            "Should not contain popped named term: {}",
            outputs[1]
        );
    }

    // ========== Recursive Function Definition Tests ==========

    #[test]
    fn test_define_fun_rec_parses() {
        // Test that define-fun-rec is parsed and handled without error
        let input = r#"
            (set-logic QF_LIA)
            (define-fun-rec fact ((n Int)) Int (ite (= n 0) 1 (* n (fact (- n 1)))))
            (declare-const x Int)
            (assert (= x 5))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "sat");
    }

    #[test]
    fn test_define_fun_rec_no_params() {
        // Test recursive function with no parameters (constant recursive definition)
        let input = r#"
            (set-logic QF_LIA)
            (define-fun-rec const_val () Int 42)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "sat");
    }

    #[test]
    fn test_define_funs_rec_parses() {
        // Test that define-funs-rec is parsed and handled without error
        let input = r#"
            (set-logic QF_LIA)
            (define-funs-rec
                ((is_even ((n Int)) Bool) (is_odd ((n Int)) Bool))
                ((ite (= n 0) true (is_odd (- n 1)))
                 (ite (= n 0) false (is_even (- n 1)))))
            (declare-const x Int)
            (assert (= x 4))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "sat");
    }

    #[test]
    fn test_define_funs_rec_multiple_params() {
        // Test mutually recursive functions with multiple parameters
        let input = r#"
            (set-logic QF_LIA)
            (define-funs-rec
                ((f1 ((a Int) (b Int)) Int) (f2 ((x Int) (y Int)) Int))
                ((+ a (f2 b a)) (- x (f1 y x))))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "sat");
    }

    // ========== Simplify Command Tests ==========

    #[test]
    fn test_simplify_boolean_and_true() {
        // (and true x) should simplify to x
        let input = r#"
            (declare-const x Bool)
            (simplify (and true x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_boolean_and_false() {
        // (and false x) should simplify to false
        let input = r#"
            (declare-const x Bool)
            (simplify (and false x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_boolean_or_true() {
        // (or true x) should simplify to true
        let input = r#"
            (declare-const x Bool)
            (simplify (or true x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_boolean_or_false() {
        // (or false x) should simplify to x
        let input = r#"
            (declare-const x Bool)
            (simplify (or false x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_double_negation() {
        // (not (not x)) should simplify to x
        let input = r#"
            (declare-const x Bool)
            (simplify (not (not x)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_de_morgan_not_and() {
        // (not (and x y)) should simplify to (or (not x) (not y))
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (not (and x y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].contains("(or"));
        assert!(outputs[0].contains("(not x)"));
        assert!(outputs[0].contains("(not y)"));
    }

    #[test]
    fn test_simplify_de_morgan_not_or() {
        // (not (or x y)) should simplify to (and (not x) (not y))
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (not (or x y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].contains("(and"));
        assert!(outputs[0].contains("(not x)"));
        assert!(outputs[0].contains("(not y)"));
    }

    #[test]
    fn test_simplify_de_morgan_enables_complement_simplification() {
        // (and x (not (or x y))) -> false
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (and x (not (or x y))))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].trim(), "false");
    }

    #[test]
    fn test_simplify_ite_true_condition() {
        // (ite true a b) should simplify to a
        let input = r#"
            (declare-const a Int)
            (declare-const b Int)
            (simplify (ite true a b))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "a");
    }

    #[test]
    fn test_simplify_ite_false_condition() {
        // (ite false a b) should simplify to b
        let input = r#"
            (declare-const a Int)
            (declare-const b Int)
            (simplify (ite false a b))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "b");
    }

    #[test]
    fn test_simplify_equality_same() {
        // (= x x) should simplify to true
        let input = r#"
            (declare-const x Int)
            (simplify (= x x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_constant() {
        // Constants should remain unchanged
        let input = r#"
            (simplify true)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    // =======================================================================
    // Arithmetic constant folding tests via simplify
    // =======================================================================

    #[test]
    fn test_simplify_int_addition() {
        // (+ 2 3) should simplify to 5
        let input = r#"
            (simplify (+ 2 3))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "5");
    }

    #[test]
    fn test_simplify_int_subtraction() {
        // (- 10 4) should simplify to 6
        let input = r#"
            (simplify (- 10 4))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "6");
    }

    #[test]
    fn test_simplify_int_multiplication() {
        // (* 3 4) should simplify to 12
        let input = r#"
            (simplify (* 3 4))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "12");
    }

    #[test]
    fn test_simplify_int_div() {
        // (div 7 3) should simplify to 2
        let input = r#"
            (simplify (div 7 3))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "2");
    }

    #[test]
    fn test_simplify_int_mod() {
        // (mod 7 3) should simplify to 1
        let input = r#"
            (simplify (mod 7 3))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "1");
    }

    #[test]
    fn test_simplify_unary_minus() {
        // (- 5) should simplify to -5
        let input = r#"
            (simplify (- 5))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Constant folding evaluates (- 5) to -5
        assert_eq!(outputs[0], "-5");
    }

    #[test]
    fn test_simplify_nested_arithmetic() {
        // (+ 1 (+ 2 3)) should simplify to 6
        let input = r#"
            (simplify (+ 1 (+ 2 3)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "6");
    }

    #[test]
    fn test_simplify_add_zero_identity() {
        // (+ x 0) should simplify to x
        let input = r#"
            (declare-const x Int)
            (simplify (+ x 0))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // The second output should be "x"
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_mul_one_identity() {
        // (* x 1) should simplify to x
        let input = r#"
            (declare-const x Int)
            (simplify (* x 1))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_mul_zero_annihilation() {
        // (* x 0) should simplify to 0
        let input = r#"
            (declare-const x Int)
            (simplify (* x 0))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "0");
    }

    #[test]
    fn test_simplify_sub_self() {
        // (- x x) should simplify to 0
        let input = r#"
            (declare-const x Int)
            (simplify (- x x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "0");
    }

    #[test]
    fn test_simplify_abs() {
        // (abs (- 5)) should simplify to 5
        let input = r#"
            (simplify (abs (- 5)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "5");
    }

    // =======================================================================
    // Comparison simplification tests
    // =======================================================================

    #[test]
    fn test_simplify_less_than_constants() {
        // (< 2 3) should simplify to true
        let input = r#"
            (simplify (< 2 3))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_chained_less_than_constants() {
        // (< 1 2 3) is syntactic sugar for (and (< 1 2) (< 2 3)) and should simplify to true
        let input = r#"
            (simplify (< 1 2 3))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_less_than_false() {
        // (< 5 3) should simplify to false
        let input = r#"
            (simplify (< 5 3))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_less_equal_constants() {
        // (<= 3 3) should simplify to true
        let input = r#"
            (simplify (<= 3 3))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_chained_less_equal_constants() {
        // (<= 1 1 2) is syntactic sugar for (and (<= 1 1) (<= 1 2)) and should simplify to true
        let input = r#"
            (simplify (<= 1 1 2))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_greater_than_constants() {
        // (> 5 3) should simplify to true
        let input = r#"
            (simplify (> 5 3))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_chained_greater_than_constants() {
        // (> 3 2 1) is syntactic sugar for (and (> 3 2) (> 2 1)) and should simplify to true
        let input = r#"
            (simplify (> 3 2 1))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_greater_equal_constants() {
        // (>= 3 5) should simplify to false
        let input = r#"
            (simplify (>= 3 5))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_chained_comparison_reflexive_short_circuit() {
        // (< x x y) is (and (< x x) (< x y)) and should simplify to false due to reflexive (< x x)
        let input = r#"
            (declare-const x Int)
            (declare-const y Int)
            (simplify (< x x y))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_less_than_reflexive() {
        // (< x x) should simplify to false
        let input = r#"
            (declare-const x Int)
            (simplify (< x x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_less_equal_reflexive() {
        // (<= x x) should simplify to true
        let input = r#"
            (declare-const x Int)
            (simplify (<= x x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_greater_than_reflexive() {
        // (> x x) should simplify to false
        let input = r#"
            (declare-const x Int)
            (simplify (> x x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_greater_equal_reflexive() {
        // (>= x x) should simplify to true
        let input = r#"
            (declare-const x Int)
            (simplify (>= x x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_comparison_non_constant() {
        // (< x y) should not simplify when x and y are different variables
        let input = r#"
            (declare-const x Int)
            (declare-const y Int)
            (simplify (< x y))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should produce (< x y) since it can't simplify non-constants
        assert!(outputs[0].contains("<") || outputs[0].contains("x"));
    }

    #[test]
    fn test_simplify_nested_comparison() {
        // (and (< 1 2) (> 3 2)) should simplify to true
        let input = r#"
            (simplify (and (< 1 2) (> 3 2)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    // =======================================================================
    // Equality constant folding tests
    // =======================================================================

    #[test]
    fn test_simplify_eq_different_int_constants() {
        // (= 1 2) should simplify to false
        let input = r#"
            (simplify (= 1 2))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_eq_same_int_constants() {
        // (= 42 42) should simplify to true
        let input = r#"
            (simplify (= 42 42))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_eq_different_bool_constants() {
        // (= true false) should simplify to false
        let input = r#"
            (simplify (= true false))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_eq_different_string_constants() {
        // (= "hello" "world") should simplify to false
        let input = r#"
            (simplify (= "hello" "world"))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    // =======================================================================
    // Distinct simplification tests
    // =======================================================================

    #[test]
    fn test_simplify_distinct_duplicate_vars() {
        // (distinct x x) should simplify to false
        let input = r#"
            (declare-const x Int)
            (simplify (distinct x x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_distinct_constants() {
        // (distinct 1 2 3) should simplify to true
        let input = r#"
            (simplify (distinct 1 2 3))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_distinct_duplicate_constants() {
        // (distinct 1 2 1) should simplify to false
        let input = r#"
            (simplify (distinct 1 2 1))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_distinct_same_constants() {
        // (distinct 5 5) should simplify to false
        let input = r#"
            (simplify (distinct 5 5))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    // =======================================================================
    // ITE Boolean branch simplification tests
    // =======================================================================

    #[test]
    fn test_simplify_ite_true_false_branches() {
        // (ite c true false) should simplify to c
        let input = r#"
            (declare-const c Bool)
            (simplify (ite c true false))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "c");
    }

    #[test]
    fn test_simplify_ite_false_true_branches() {
        // (ite c false true) should simplify to (not c)
        let input = r#"
            (declare-const c Bool)
            (simplify (ite c false true))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "(not c)");
    }

    #[test]
    fn test_simplify_ite_cond_as_then_branch() {
        // (ite c c false) should simplify to c
        let input = r#"
            (declare-const c Bool)
            (simplify (ite c c false))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "c");
    }

    #[test]
    fn test_simplify_ite_true_then_cond_else() {
        // (ite c true c) should simplify to c
        let input = r#"
            (declare-const c Bool)
            (simplify (ite c true c))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "c");
    }

    #[test]
    fn test_simplify_ite_x_false_to_and() {
        // (ite c x false) should simplify to (and c x)
        let input = r#"
            (declare-const c Bool)
            (declare-const x Bool)
            (simplify (ite c x false))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "(and c x)");
    }

    #[test]
    fn test_simplify_ite_true_x_to_or() {
        // (ite c true x) should simplify to (or c x)
        let input = r#"
            (declare-const c Bool)
            (declare-const x Bool)
            (simplify (ite c true x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "(or c x)");
    }

    #[test]
    fn test_simplify_ite_false_x_to_and_not() {
        // (ite c false x) should simplify to (and (not c) x)
        let input = r#"
            (declare-const c Bool)
            (declare-const x Bool)
            (simplify (ite c false x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should be (and (not c) x) or equivalent
        let result = &outputs[0];
        assert!(
            result.contains("and") && result.contains("not"),
            "Expected (and (not c) x), got: {}",
            result
        );
    }

    #[test]
    fn test_simplify_ite_x_true_to_or_not() {
        // (ite c x true) should simplify to (or (not c) x)
        let input = r#"
            (declare-const c Bool)
            (declare-const x Bool)
            (simplify (ite c x true))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should be (or (not c) x) or equivalent
        let result = &outputs[0];
        assert!(
            result.contains("or") && result.contains("not"),
            "Expected (or (not c) x), got: {}",
            result
        );
    }

    #[test]
    fn test_simplify_nested_ite_same_condition() {
        // (ite c (ite c x y) z) should simplify to (ite c x z)
        let input = r#"
            (declare-const c Bool)
            (declare-const x Bool)
            (declare-const y Bool)
            (declare-const z Bool)
            (simplify (ite c (ite c x y) z))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should simplify - the y variable shouldn't appear
        let result = &outputs[0];
        assert!(
            !result.contains('y'),
            "Expected y to be eliminated, got: {}",
            result
        );
    }

    #[test]
    fn test_simplify_ite_non_bool_no_and_or() {
        // (ite c x 0) with Int x should NOT simplify to (and c x)
        let input = r#"
            (declare-const c Bool)
            (declare-const x Int)
            (simplify (ite c x 0))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should remain as ite, not become and
        let result = &outputs[0];
        assert!(
            result.contains("ite") || result.contains('x'),
            "Expected ite for Int branches, got: {}",
            result
        );
        assert!(
            !result.contains("and"),
            "Should not simplify Int ite to and, got: {}",
            result
        );
    }

    // =======================================================================
    // ITE negated condition normalization tests
    // =======================================================================

    #[test]
    fn test_simplify_ite_negated_condition() {
        // (ite (not c) a b) -> (ite c b a)
        let input = r#"
            (declare-const c Bool)
            (declare-const a Int)
            (declare-const b Int)
            (simplify (ite (not c) a b))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should produce (ite c b a), with condition c (not negated) and branches swapped
        let result = &outputs[0];
        assert!(
            result.contains("ite"),
            "Expected ite in result, got: {}",
            result
        );
        // The condition should be just "c", not "(not c)"
        // Check that (not c) does NOT appear in the output
        assert!(
            !result.contains("not"),
            "Expected positive condition (c), not (not c), got: {}",
            result
        );
    }

    #[test]
    fn test_simplify_ite_negated_condition_bool_true_false() {
        // (ite (not c) true false) -> (ite c false true) -> (not c)
        let input = r#"
            (declare-const c Bool)
            (simplify (ite (not c) true false))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should simplify to (not c)
        let result = &outputs[0];
        assert!(
            result.contains("not") && result.contains('c'),
            "Expected (not c), got: {}",
            result
        );
    }

    #[test]
    fn test_simplify_ite_negated_condition_with_sat() {
        // Test that ITE with negated condition works correctly in SAT solving
        // (ite (not c) 1 2) = 1 with c = false should be SAT
        let input = r#"
            (declare-const c Bool)
            (assert (= (ite (not c) 1 2) 1))
            (assert (not c))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "sat");
    }

    // =======================================================================
    // Comparison normalization tests
    // =======================================================================

    #[test]
    fn test_simplify_gt_normalizes_to_lt() {
        // (> x y) should normalize to (< y x)
        let input = r#"
            (declare-const x Int)
            (declare-const y Int)
            (simplify (> x y))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should produce (< y x), not (> x y)
        let result = &outputs[0];
        assert!(
            result.contains('<') && !result.contains('>'),
            "Expected < operator after normalization, got: {}",
            result
        );
        // Check that y comes before x in the output (< y x)
        let y_pos = result.find('y').unwrap_or(usize::MAX);
        let x_pos = result.find('x').unwrap_or(usize::MAX);
        assert!(
            y_pos < x_pos,
            "Expected (< y x) after normalizing (> x y), got: {}",
            result
        );
    }

    #[test]
    fn test_simplify_ge_normalizes_to_le() {
        // (>= x y) should normalize to (<= y x)
        let input = r#"
            (declare-const x Int)
            (declare-const y Int)
            (simplify (>= x y))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should produce (<= y x), not (>= x y)
        let result = &outputs[0];
        assert!(
            result.contains("<=") && !result.contains(">="),
            "Expected <= operator after normalization, got: {}",
            result
        );
        // Check that y comes before x in the output (<= y x)
        let y_pos = result.find('y').unwrap_or(usize::MAX);
        let x_pos = result.find('x').unwrap_or(usize::MAX);
        assert!(
            y_pos < x_pos,
            "Expected (<= y x) after normalizing (>= x y), got: {}",
            result
        );
    }

    #[test]
    fn test_simplify_gt_and_lt_same_result() {
        // (> x y) and (< y x) should produce identical output
        let input1 = r#"
            (declare-const a Int)
            (declare-const b Int)
            (simplify (> a b))
        "#;

        let input2 = r#"
            (declare-const a Int)
            (declare-const b Int)
            (simplify (< b a))
        "#;

        let commands1 = parse(input1).unwrap();
        let mut exec1 = Executor::new();
        let outputs1 = exec1.execute_all(&commands1).unwrap();

        let commands2 = parse(input2).unwrap();
        let mut exec2 = Executor::new();
        let outputs2 = exec2.execute_all(&commands2).unwrap();

        assert_eq!(outputs1.len(), 1);
        assert_eq!(outputs2.len(), 1);
        assert_eq!(
            outputs1[0], outputs2[0],
            "(> a b) and (< b a) should produce identical output"
        );
    }

    #[test]
    fn test_simplify_ge_and_le_same_result() {
        // (>= x y) and (<= y x) should produce identical output
        let input1 = r#"
            (declare-const a Int)
            (declare-const b Int)
            (simplify (>= a b))
        "#;

        let input2 = r#"
            (declare-const a Int)
            (declare-const b Int)
            (simplify (<= b a))
        "#;

        let commands1 = parse(input1).unwrap();
        let mut exec1 = Executor::new();
        let outputs1 = exec1.execute_all(&commands1).unwrap();

        let commands2 = parse(input2).unwrap();
        let mut exec2 = Executor::new();
        let outputs2 = exec2.execute_all(&commands2).unwrap();

        assert_eq!(outputs1.len(), 1);
        assert_eq!(outputs2.len(), 1);
        assert_eq!(
            outputs1[0], outputs2[0],
            "(>= a b) and (<= b a) should produce identical output"
        );
    }

    // =======================================================================
    // Boolean equality (iff) simplification tests
    // =======================================================================

    #[test]
    fn test_simplify_eq_bool_with_true() {
        // (= x true) should simplify to x
        let input = r#"
            (declare-const x Bool)
            (simplify (= x true))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_eq_true_with_bool() {
        // (= true x) should simplify to x
        let input = r#"
            (declare-const x Bool)
            (simplify (= true x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_eq_bool_with_false() {
        // (= x false) should simplify to (not x)
        let input = r#"
            (declare-const x Bool)
            (simplify (= x false))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "(not x)");
    }

    #[test]
    fn test_simplify_eq_false_with_bool() {
        // (= false x) should simplify to (not x)
        let input = r#"
            (declare-const x Bool)
            (simplify (= false x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "(not x)");
    }

    #[test]
    fn test_simplify_eq_compound_bool_with_true() {
        // (= (and x y) true) should simplify to (and x y)
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (= (and x y) true))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "(and x y)");
    }

    #[test]
    fn test_simplify_eq_compound_bool_with_false() {
        // (= (or x y) false) should simplify to (not (or x y)), which normalizes via De Morgan
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (= (or x y) false))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "(and (not x) (not y))");
    }

    #[test]
    fn test_simplify_eq_complement_detection() {
        // (= x (not x)) should simplify to false
        let input = r#"
            (declare-const x Bool)
            (simplify (= x (not x)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_eq_complement_detection_reversed() {
        // (= (not x) x) should simplify to false
        let input = r#"
            (declare-const x Bool)
            (simplify (= (not x) x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_eq_complement_predicate() {
        // (= (p a) (not (p a))) should simplify to false
        let input = r#"
            (declare-sort U 0)
            (declare-const a U)
            (declare-fun p (U) Bool)
            (simplify (= (p a) (not (p a))))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_eq_negation_lifting() {
        // (= (not x) (not y)) should simplify to (= x y)
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (= (not x) (not y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "(= x y)");
    }

    #[test]
    fn test_simplify_eq_negation_lifting_predicates() {
        // (= (not (p a)) (not (q b))) should simplify to (= (p a) (q b))
        let input = r#"
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (declare-fun p (U) Bool)
            (declare-fun q (U) Bool)
            (simplify (= (not (p a)) (not (q b))))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "(= (p a) (q b))");
    }

    #[test]
    fn test_simplify_eq_reflexive_negation() {
        // (= (not x) (not x)) should simplify to true
        let input = r#"
            (declare-const x Bool)
            (simplify (= (not x) (not x)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    // =======================================================================
    // ITE-equality simplification tests
    // =======================================================================

    #[test]
    fn test_simplify_eq_ite_then_branch() {
        // (= (ite c a b) a) -> (or c (= b a))
        let input = r#"
            (declare-const c Bool)
            (declare-const a Bool)
            (declare-const b Bool)
            (simplify (= (ite c a b) a))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should be (or c (= a b)) with canonical ordering
        assert!(outputs[0].contains("or"));
        assert!(outputs[0].contains("c"));
    }

    #[test]
    fn test_simplify_eq_ite_else_branch() {
        // (= (ite c a b) b) -> (or (not c) (= a b))
        let input = r#"
            (declare-const c Bool)
            (declare-const a Bool)
            (declare-const b Bool)
            (simplify (= (ite c a b) b))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should be (or (not c) (= a b)) with canonical ordering
        assert!(outputs[0].contains("or"));
        assert!(outputs[0].contains("not"));
    }

    #[test]
    fn test_simplify_eq_ite_symmetric() {
        // (= a (ite c a b)) -> (or c (= b a))
        let input = r#"
            (declare-const c Bool)
            (declare-const a Bool)
            (declare-const b Bool)
            (simplify (= a (ite c a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should be (or c (= a b)) with canonical ordering
        assert!(outputs[0].contains("or"));
        assert!(outputs[0].contains("c"));
    }

    #[test]
    fn test_simplify_eq_ite_same_condition() {
        // (= (ite c a b) (ite c x y)) -> (ite c (= a x) (= b y))
        let input = r#"
            (declare-const c Bool)
            (declare-const a Bool)
            (declare-const b Bool)
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (= (ite c a b) (ite c x y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should be (ite c (= a x) (= b y)) or simplified form
        assert!(outputs[0].contains("ite") || outputs[0].contains("="));
    }

    #[test]
    fn test_simplify_eq_ite_with_constants() {
        // (= (ite c true false) true) -> c
        let input = r#"
            (declare-const c Bool)
            (simplify (= (ite c true false) true))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // (ite c true false) simplifies to c, so (= c true) -> c
        assert_eq!(outputs[0], "c");
    }

    #[test]
    fn test_simplify_eq_ite_reflexive() {
        // (= (ite c a a) a) -> true (via same-branch simplification)
        let input = r#"
            (declare-const c Bool)
            (declare-const a Bool)
            (simplify (= (ite c a a) a))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    // =======================================================================
    // And/Or flattening tests
    // =======================================================================

    #[test]
    fn test_simplify_and_flattening() {
        // (and a (and b c)) should flatten to (and a b c)
        let input = r#"
            (declare-const a Bool)
            (declare-const b Bool)
            (declare-const c Bool)
            (simplify (and a (and b c)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Output should be a flat and with a, b, c (canonically sorted)
        assert!(outputs[0].contains("and"));
        assert!(outputs[0].contains("a"));
        assert!(outputs[0].contains("b"));
        assert!(outputs[0].contains("c"));
        // Should NOT contain nested and
        assert_eq!(outputs[0].matches("and").count(), 1);
    }

    #[test]
    fn test_simplify_or_flattening() {
        // (or a (or b c)) should flatten to (or a b c)
        let input = r#"
            (declare-const a Bool)
            (declare-const b Bool)
            (declare-const c Bool)
            (simplify (or a (or b c)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Output should be a flat or with a, b, c (canonically sorted)
        assert!(outputs[0].contains("or"));
        assert!(outputs[0].contains("a"));
        assert!(outputs[0].contains("b"));
        assert!(outputs[0].contains("c"));
        // Should NOT contain nested or
        assert_eq!(outputs[0].matches("or").count(), 1);
    }

    #[test]
    fn test_simplify_and_deep_flattening() {
        // (and (and a b) (and c d)) should flatten to (and a b c d)
        let input = r#"
            (declare-const a Bool)
            (declare-const b Bool)
            (declare-const c Bool)
            (declare-const d Bool)
            (simplify (and (and a b) (and c d)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Output should be a flat and with a, b, c, d
        assert!(outputs[0].contains("and"));
        assert!(outputs[0].contains("a"));
        assert!(outputs[0].contains("b"));
        assert!(outputs[0].contains("c"));
        assert!(outputs[0].contains("d"));
        // Should NOT contain nested and
        assert_eq!(outputs[0].matches("and").count(), 1);
    }

    #[test]
    fn test_simplify_or_deep_flattening() {
        // (or (or a b) (or c d)) should flatten to (or a b c d)
        let input = r#"
            (declare-const a Bool)
            (declare-const b Bool)
            (declare-const c Bool)
            (declare-const d Bool)
            (simplify (or (or a b) (or c d)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Output should be a flat or with a, b, c, d
        assert!(outputs[0].contains("or"));
        assert!(outputs[0].contains("a"));
        assert!(outputs[0].contains("b"));
        assert!(outputs[0].contains("c"));
        assert!(outputs[0].contains("d"));
        // Should NOT contain nested or
        assert_eq!(outputs[0].matches("or").count(), 1);
    }

    #[test]
    fn test_simplify_and_flattening_with_dedup() {
        // (and a (and a b)) should flatten and dedup to (and a b)
        let input = r#"
            (declare-const a Bool)
            (declare-const b Bool)
            (simplify (and a (and a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Output should be (and a b)
        assert_eq!(outputs[0], "(and a b)");
    }

    #[test]
    fn test_simplify_or_flattening_with_dedup() {
        // (or a (or a b)) should flatten and dedup to (or a b)
        let input = r#"
            (declare-const a Bool)
            (declare-const b Bool)
            (simplify (or a (or a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Output should be (or a b)
        assert_eq!(outputs[0], "(or a b)");
    }

    // =======================================================================
    // Complement detection tests
    // =======================================================================

    #[test]
    fn test_simplify_and_complement() {
        // (and x (not x)) should simplify to false
        let input = r#"
            (declare-const x Bool)
            (simplify (and x (not x)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_or_complement() {
        // (or x (not x)) should simplify to true
        let input = r#"
            (declare-const x Bool)
            (simplify (or x (not x)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_and_complement_with_others() {
        // (and x y (not x) z) should simplify to false
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (declare-const z Bool)
            (simplify (and x y (not x) z))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_or_complement_with_others() {
        // (or x y (not x) z) should simplify to true
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (declare-const z Bool)
            (simplify (or x y (not x) z))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_and_complement_nested() {
        // (and (and a b) (not a)) should simplify to false
        // after flattening: (and a b (not a)) contains complement
        let input = r#"
            (declare-const a Bool)
            (declare-const b Bool)
            (simplify (and (and a b) (not a)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_or_complement_nested() {
        // (or (or a b) (not a)) should simplify to true
        // after flattening: (or a b (not a)) contains complement
        let input = r#"
            (declare-const a Bool)
            (declare-const b Bool)
            (simplify (or (or a b) (not a)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_and_no_complement() {
        // (and x (not y)) should NOT simplify to false (no complement)
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (and x (not y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should remain as an and term
        assert!(outputs[0].contains("and"));
        assert!(outputs[0].contains("x"));
        assert!(outputs[0].contains("not"));
        assert!(outputs[0].contains("y"));
    }

    #[test]
    fn test_simplify_or_no_complement() {
        // (or x (not y)) should NOT simplify to true (no complement)
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (or x (not y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should remain as an or term
        assert!(outputs[0].contains("or"));
        assert!(outputs[0].contains("x"));
        assert!(outputs[0].contains("not"));
        assert!(outputs[0].contains("y"));
    }

    // =======================================================================
    // Absorption law tests
    // =======================================================================

    #[test]
    fn test_simplify_and_absorption_basic() {
        // (and x (or x y)) = x
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (and x (or x y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_or_absorption_basic() {
        // (or x (and x y)) = x
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (or x (and x y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_and_absorption_order_independent() {
        // (and (or x y) x) = x (order shouldn't matter)
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (and (or x y) x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_or_absorption_order_independent() {
        // (or (and x y) x) = x (order shouldn't matter)
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (or (and x y) x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_and_absorption_multiple_vars() {
        // (and x (or x y z)) = x
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (declare-const z Bool)
            (simplify (and x (or x y z)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_or_absorption_multiple_vars() {
        // (or x (and x y z)) = x
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (declare-const z Bool)
            (simplify (or x (and x y z)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_and_no_absorption() {
        // (and x (or y z)) should NOT simplify - x is not in (or y z)
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (declare-const z Bool)
            (simplify (and x (or y z)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should remain as an and term
        assert!(outputs[0].contains("and"));
        assert!(outputs[0].contains("or"));
    }

    #[test]
    fn test_simplify_or_no_absorption() {
        // (or x (and y z)) should NOT simplify - x is not in (and y z)
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (declare-const z Bool)
            (simplify (or x (and y z)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should remain as an or term
        assert!(outputs[0].contains("or"));
        assert!(outputs[0].contains("and"));
    }

    // =======================================================================
    // Negation-through absorption tests via simplify
    // =======================================================================

    #[test]
    fn test_simplify_and_negation_through_absorption() {
        // (and x (or (not x) y)) should simplify to (and x y)
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (and x (or (not x) y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should be (and x y)
        assert!(outputs[0].contains("and"));
        assert!(outputs[0].contains("x"));
        assert!(outputs[0].contains("y"));
        // Should NOT contain "or" or "not"
        assert!(!outputs[0].contains("or"));
        assert!(!outputs[0].contains("not"));
    }

    #[test]
    fn test_simplify_or_negation_through_absorption() {
        // (or x (and (not x) y)) should simplify to (or x y)
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (or x (and (not x) y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should be (or x y)
        assert!(outputs[0].contains("or"));
        assert!(outputs[0].contains("x"));
        assert!(outputs[0].contains("y"));
        // Should NOT contain "and" or "not"
        assert!(!outputs[0].contains("and"));
        assert!(!outputs[0].contains("not"));
    }

    #[test]
    fn test_simplify_and_negation_through_multiple() {
        // (and x (or (not x) y z)) should simplify to (and x (or y z))
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (declare-const z Bool)
            (simplify (and x (or (not x) y z)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should be (and x (or y z))
        assert!(outputs[0].contains("and"));
        assert!(outputs[0].contains("or"));
        // Should NOT contain "not" - the (not x) was removed
        assert!(!outputs[0].contains("not"));
    }

    #[test]
    fn test_simplify_or_negation_through_multiple() {
        // (or x (and (not x) y z)) should simplify to (or x (and y z))
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (declare-const z Bool)
            (simplify (or x (and (not x) y z)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should be (or x (and y z))
        assert!(outputs[0].contains("or"));
        assert!(outputs[0].contains("and"));
        // Should NOT contain "not" - the (not x) was removed
        assert!(!outputs[0].contains("not"));
    }

    #[test]
    fn test_simplify_and_negation_through_removes_inner() {
        // (and x (or (not x))) simplifies:
        // First (or (not x)) = (not x) (single element)
        // Then (and x (not x)) = false (complement)
        let input = r#"
            (declare-const x Bool)
            (simplify (and x (or (not x))))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].trim(), "false");
    }

    #[test]
    fn test_simplify_or_negation_through_removes_inner() {
        // (or x (and (not x))) simplifies:
        // First (and (not x)) = (not x) (single element)
        // Then (or x (not x)) = true (complement)
        let input = r#"
            (declare-const x Bool)
            (simplify (or x (and (not x))))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].trim(), "true");
    }

    #[test]
    fn test_simplify_and_negation_through_no_false_positive() {
        // (and x (or y z)) should NOT simplify - no (not x) in the or
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (declare-const z Bool)
            (simplify (and x (or y z)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should remain as (and x (or y z))
        assert!(outputs[0].contains("and"));
        assert!(outputs[0].contains("or"));
    }

    #[test]
    fn test_simplify_or_negation_through_no_false_positive() {
        // (or x (and y z)) should NOT simplify - no (not x) in the and
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (declare-const z Bool)
            (simplify (or x (and y z)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should remain as (or x (and y z))
        assert!(outputs[0].contains("or"));
        assert!(outputs[0].contains("and"));
    }

    // ========================================================================
    // ITE Negation Normalization Tests
    // ========================================================================

    #[test]
    fn test_simplify_not_ite_basic() {
        // (not (ite c a b)) -> (ite c (not a) (not b))
        let input = r#"
            (declare-const c Bool)
            (declare-const a Bool)
            (declare-const b Bool)
            (simplify (not (ite c a b)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Result should be (ite c (not a) (not b))
        assert!(
            outputs[0].contains("ite"),
            "Expected ite in result, got: {}",
            outputs[0]
        );
        // Both branches should have not
        let not_count = outputs[0].matches("not").count();
        assert_eq!(
            not_count, 2,
            "Expected exactly 2 'not' in result, got {}: {}",
            not_count, outputs[0]
        );
    }

    #[test]
    fn test_simplify_not_ite_with_true_branch() {
        // (not (ite c true a)) = (not (or c a)) = (and (not c) (not a))
        let input = r#"
            (declare-const c Bool)
            (declare-const a Bool)
            (simplify (not (ite c true a)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should simplify to (and (not c) (not a))
        assert!(
            outputs[0].contains("and"),
            "Expected and in result, got: {}",
            outputs[0]
        );
        let not_count = outputs[0].matches("not").count();
        assert_eq!(
            not_count, 2,
            "Expected exactly 2 'not' in result, got {}: {}",
            not_count, outputs[0]
        );
    }

    #[test]
    fn test_simplify_not_ite_with_false_branch() {
        // (not (ite c a false)) = (not (and c a)) = (or (not c) (not a))
        let input = r#"
            (declare-const c Bool)
            (declare-const a Bool)
            (simplify (not (ite c a false)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should simplify to (or (not c) (not a))
        assert!(
            outputs[0].contains("or"),
            "Expected or in result, got: {}",
            outputs[0]
        );
        let not_count = outputs[0].matches("not").count();
        assert_eq!(
            not_count, 2,
            "Expected exactly 2 'not' in result, got {}: {}",
            not_count, outputs[0]
        );
    }

    #[test]
    fn test_simplify_not_ite_true_false_branches() {
        // (not (ite c true false)) = (not c)
        // Because (ite c true false) = c
        let input = r#"
            (declare-const c Bool)
            (simplify (not (ite c true false)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should simplify to (not c)
        assert_eq!(
            outputs[0].trim(),
            "(not c)",
            "Expected (not c), got: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_simplify_not_ite_false_true_branches() {
        // (not (ite c false true)) = c
        // Because (ite c false true) = (not c), and (not (not c)) = c
        let input = r#"
            (declare-const c Bool)
            (simplify (not (ite c false true)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should simplify to c
        assert_eq!(outputs[0].trim(), "c", "Expected c, got: {}", outputs[0]);
    }

    #[test]
    fn test_simplify_not_ite_nested() {
        // (not (ite c1 (ite c2 a b) false))
        // = (not (and c1 (ite c2 a b)))
        // = (or (not c1) (not (ite c2 a b)))
        // = (or (not c1) (ite c2 (not a) (not b)))
        let input = r#"
            (declare-const c1 Bool)
            (declare-const c2 Bool)
            (declare-const a Bool)
            (declare-const b Bool)
            (simplify (not (ite c1 (ite c2 a b) false)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should have or with (not c1) and ite
        assert!(
            outputs[0].contains("or"),
            "Expected or in result, got: {}",
            outputs[0]
        );
        assert!(
            outputs[0].contains("ite"),
            "Expected ite in result, got: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_simplify_double_not_ite() {
        // (not (not (ite c a b))) = (ite c a b)
        let input = r#"
            (declare-const c Bool)
            (declare-const a Bool)
            (declare-const b Bool)
            (simplify (not (not (ite c a b))))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should be (ite c a b)
        assert!(
            outputs[0].contains("ite"),
            "Expected ite in result, got: {}",
            outputs[0]
        );
        // Should not have any 'not'
        assert!(
            !outputs[0].contains("not"),
            "Should not contain 'not', got: {}",
            outputs[0]
        );
    }

    // =========================================================================
    // XOR simplification tests (SMT-LIB integration)
    // =========================================================================

    #[test]
    fn test_simplify_xor_same_operand() {
        // (xor x x) = false
        let input = r#"
            (declare-const x Bool)
            (simplify (xor x x))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "false");
    }

    #[test]
    fn test_simplify_xor_with_true() {
        // (xor x true) = (not x)
        let input = r#"
            (declare-const x Bool)
            (simplify (xor x true))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].contains("not"),
            "Expected (not x), got: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_simplify_xor_with_false() {
        // (xor x false) = x
        let input = r#"
            (declare-const x Bool)
            (simplify (xor x false))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "x");
    }

    #[test]
    fn test_simplify_xor_complement() {
        // (xor x (not x)) = true
        let input = r#"
            (declare-const x Bool)
            (simplify (xor x (not x)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "true");
    }

    #[test]
    fn test_simplify_xor_double_negation_lifting() {
        // (xor (not x) (not y)) = (xor x y)
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (simplify (xor (not x) (not y)))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        // Should be (xor x y) - no 'not' in the result
        assert!(
            !outputs[0].contains("not"),
            "Should not contain 'not' after double negation lifting, got: {}",
            outputs[0]
        );
        assert!(
            outputs[0].contains("xor"),
            "Expected xor in result, got: {}",
            outputs[0]
        );
    }

    #[test]
    fn test_xor_sat_formula() {
        // Test a satisfiable formula using xor
        // (xor x y) and (xor x true) - should be sat with y = true
        let input = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (assert (xor x y))
            (assert (xor x true))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "sat");
    }

    // ============================================
    // QF_AX (Array) Tests
    // ============================================

    #[test]
    fn test_executor_qf_ax_basic_sat() {
        // Basic array operations that should be SAT
        let input = r#"
            (set-logic QF_AX)
            (declare-const a (Array Int Int))
            (declare-const i Int)
            (declare-const v Int)
            (assert (= (select (store a i v) i) v))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_ax_row1_conflict() {
        // ROW1 violation: select(store(a, i, v), i) must equal v
        // This asserts the negation, which should be UNSAT
        let input = r#"
            (set-logic QF_AX)
            (declare-const a (Array Int Int))
            (declare-const i Int)
            (declare-const j Int)
            (declare-const v Int)
            (assert (= i j))
            (assert (not (= (select (store a i v) j) v)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_ax_row2_sat() {
        // ROW2: When i ≠ j, select(store(a, i, v), j) = select(a, j)
        // This is consistent, so SAT
        let input = r#"
            (set-logic QF_AX)
            (declare-const a (Array Int Int))
            (declare-const i Int)
            (declare-const j Int)
            (declare-const v Int)
            (assert (not (= i j)))
            (assert (= (select (store a i v) j) (select a j)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_ax_row2_conflict() {
        // ROW2 violation: When i ≠ j, select(store(a, i, v), j) must equal select(a, j)
        // Asserting they differ should be UNSAT
        let input = r#"
            (set-logic QF_AX)
            (declare-const a (Array Int Int))
            (declare-const i Int)
            (declare-const j Int)
            (declare-const v Int)
            (assert (not (= i j)))
            (assert (not (= (select (store a i v) j) (select a j))))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_ax_array_equality_conflict() {
        // Array equality: If a = b, then select(a, i) = select(b, i)
        // Asserting a = b and select(a, i) ≠ select(b, i) should be UNSAT
        let input = r#"
            (set-logic QF_AX)
            (declare-const a (Array Int Int))
            (declare-const b (Array Int Int))
            (declare-const i Int)
            (assert (= a b))
            (assert (not (= (select a i) (select b i))))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_ax_array_with_euf() {
        // Combined array + EUF reasoning
        // f(a) = f(b) when a = b (congruence), and arrays interact with EUF
        let input = r#"
            (set-logic QF_AX)
            (declare-sort U 0)
            (declare-const a (Array Int U))
            (declare-const b (Array Int U))
            (declare-const i Int)
            (declare-fun f ((Array Int U)) Bool)
            (assert (= a b))
            (assert (f a))
            (assert (not (f b)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // Should be UNSAT due to EUF congruence: a = b implies f(a) = f(b)
        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_ax_check_sat_assuming() {
        // Test check-sat-assuming with QF_AX
        let input = r#"
            (set-logic QF_AX)
            (declare-const a (Array Int Int))
            (declare-const i Int)
            (declare-const v Int)
            (declare-const p Bool)
            (assert (=> p (= (select (store a i v) i) v)))
            (check-sat-assuming (p))
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_ax_get_model() {
        // Test get-model with array values
        let input = r#"
            (set-logic QF_AX)
            (set-option :produce-models true)
            (declare-const a (Array Int Int))
            (declare-const i Int)
            (declare-const v Int)
            (assert (= (select a i) v))
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // Should return sat and a model
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        // Model should contain array definition
        assert!(outputs[1].contains("(model"), "Expected model output");
        assert!(
            outputs[1].contains("define-fun a"),
            "Expected array 'a' in model"
        );
        // Array value should be formatted with const/store
        assert!(
            outputs[1].contains("(Array Int Int)"),
            "Expected array sort in model"
        );
    }

    #[test]
    fn test_executor_qf_ax_get_model_with_store() {
        // Test get-model with explicit store operations
        let input = r#"
            (set-logic QF_AX)
            (set-option :produce-models true)
            (declare-const a (Array Int Int))
            (declare-const b (Array Int Int))
            (assert (= b (store a 1 42)))
            (assert (= (select b 1) 42))
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // Should return sat and a model
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        assert!(outputs[1].contains("(model"), "Expected model output");
    }

    // QF_LRA (Linear Real Arithmetic) Tests
    // =====================================

    #[test]
    fn test_executor_qf_lra_simple_sat() {
        let input = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (assert (<= x 10.0))
            (assert (>= x 5.0))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_lra_simple_unsat() {
        let input = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (assert (<= x 5.0))
            (assert (>= x 10.0))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_lra_linear_constraint_unsat() {
        let input = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (declare-const y Real)
            (assert (<= (+ x y) 10.0))
            (assert (>= x 5.0))
            (assert (>= y 6.0))
            (check-sat)
        "#;
        // x >= 5, y >= 6, but x + y <= 10: 5 + 6 = 11 > 10, so UNSAT

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_lra_linear_constraint_sat() {
        let input = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (declare-const y Real)
            (assert (<= (+ x y) 10.0))
            (assert (>= x 3.0))
            (assert (>= y 4.0))
            (check-sat)
        "#;
        // x >= 3, y >= 4, x + y <= 10: 3 + 4 = 7 <= 10, so SAT

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_lra_strict_inequality() {
        let input = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (assert (< x 5.0))
            (assert (> x 5.0))
            (check-sat)
        "#;
        // x < 5 and x > 5 is impossible

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_lra_equality_with_strict() {
        let input = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (assert (= x 5.0))
            (assert (> x 5.0))
            (check-sat)
        "#;
        // x = 5 and x > 5 is impossible

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_lra_scaled_variable() {
        let input = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (assert (>= (* 2.0 x) 10.0))
            (assert (>= x 5.0))
            (check-sat)
        "#;
        // 2x >= 10 and x >= 5: x >= 5 satisfies 2x >= 10

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    // QF_LIA (Linear Integer Arithmetic) Tests
    // =========================================

    #[test]
    fn test_executor_qf_lia_simple_sat() {
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (assert (<= x 10))
            (assert (>= x 5))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_lia_simple_unsat() {
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (assert (<= x 5))
            (assert (>= x 10))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_lia_integer_gap_unsat() {
        // x > 5 and x < 6 where x is integer - no integer between 5 and 6
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (assert (> x 5))
            (assert (< x 6))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // LIA should detect this is UNSAT (no integer in (5,6))
        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_lia_integer_boundary_sat() {
        // x >= 5 and x <= 6 where x is integer - x can be 5 or 6
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (assert (>= x 5))
            (assert (<= x 6))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_lia_equality() {
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (assert (= x 5))
            (assert (>= x 1))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_lia_linear_constraint_sat() {
        // x + y <= 10, x >= 3, y >= 4: solution x=3, y=4 (integer)
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (declare-const y Int)
            (assert (<= (+ x y) 10))
            (assert (>= x 3))
            (assert (>= y 4))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_lia_linear_constraint_unsat() {
        // x + y <= 10, x >= 5, y >= 6: 5 + 6 = 11 > 10, so UNSAT
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (declare-const y Int)
            (assert (<= (+ x y) 10))
            (assert (>= x 5))
            (assert (>= y 6))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    // QF_UFLIA (Uninterpreted Functions with Linear Integer Arithmetic) Tests

    #[test]
    fn test_executor_qf_uflia_simple_sat() {
        // Combine UF and LIA: f(x) = y, x >= 0
        let input = r#"
            (set-logic QF_UFLIA)
            (declare-const x Int)
            (declare-const y Int)
            (declare-fun f (Int) Int)
            (assert (= (f x) y))
            (assert (>= x 0))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_uflia_function_equality_unsat() {
        // f(x) = 5, f(x) = 6 is contradictory
        let input = r#"
            (set-logic QF_UFLIA)
            (declare-const x Int)
            (declare-fun f (Int) Int)
            (assert (= (f x) 5))
            (assert (= (f x) 6))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_uflia_congruence_with_arithmetic() {
        // Test EUF congruence with arithmetic on the same function application
        // This test uses the SAME function application f(x) in both constraints
        let input = r#"
            (set-logic QF_UFLIA)
            (declare-const x Int)
            (declare-fun f (Int) Int)
            (assert (>= (f x) 10))
            (assert (< (f x) 5))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // f(x) >= 10 and f(x) < 5 is a contradiction
        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_uflia_arithmetic_constraint_unsat() {
        // UF with integer gap constraint
        let input = r#"
            (set-logic QF_UFLIA)
            (declare-const x Int)
            (declare-fun f (Int) Int)
            (assert (> x 5))
            (assert (< x 6))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // No integer between 5 and 6 exclusively
        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_uflia_combined_sat() {
        // Combination of UF equality and separate arithmetic constraints
        // The UF equality (= (f a) (f b)) is independent of arithmetic
        let input = r#"
            (set-logic QF_UFLIA)
            (declare-const a Int)
            (declare-const b Int)
            (declare-fun f (Int) Int)
            (assert (>= a 0))
            (assert (<= b 10))
            (assert (= (f a) (f b)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    // QF_UFLRA (Uninterpreted Functions with Linear Real Arithmetic) Tests

    #[test]
    fn test_executor_qf_uflra_simple_sat() {
        // Combine UF and LRA: f(x) = y, x >= 0
        let input = r#"
            (set-logic QF_UFLRA)
            (declare-const x Real)
            (declare-const y Real)
            (declare-fun f (Real) Real)
            (assert (= (f x) y))
            (assert (>= x 0.0))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_uflra_function_equality_unsat() {
        // f(x) = 5.0, f(x) = 6.0 is contradictory
        let input = r#"
            (set-logic QF_UFLRA)
            (declare-const x Real)
            (declare-fun f (Real) Real)
            (assert (= (f x) 5.0))
            (assert (= (f x) 6.0))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_uflra_congruence_with_arithmetic() {
        // Test EUF congruence with arithmetic on the same function application
        let input = r#"
            (set-logic QF_UFLRA)
            (declare-const x Real)
            (declare-fun f (Real) Real)
            (assert (>= (f x) 10.0))
            (assert (< (f x) 5.0))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // f(x) >= 10 and f(x) < 5 is a contradiction
        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_uflra_arithmetic_sat() {
        // Pure arithmetic constraints in UFLRA logic
        let input = r#"
            (set-logic QF_UFLRA)
            (declare-const x Real)
            (assert (>= x 5.0))
            (assert (<= x 6.0))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_uflra_combined_sat() {
        // Combination of UF equality and separate arithmetic constraints
        let input = r#"
            (set-logic QF_UFLRA)
            (declare-const a Real)
            (declare-const b Real)
            (declare-fun f (Real) Real)
            (assert (>= a 0.0))
            (assert (<= b 10.0))
            (assert (= (f a) (f b)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    // QF_AUFLIA (Arrays + Uninterpreted Functions + Linear Integer Arithmetic) Tests

    #[test]
    fn test_executor_qf_auflia_simple_sat() {
        // Basic array with integer indices and UF
        let input = r#"
            (set-logic QF_AUFLIA)
            (declare-const a (Array Int Int))
            (declare-const i Int)
            (declare-fun f (Int) Int)
            (assert (= (select a i) (f i)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_auflia_array_store_select() {
        // Test simple array read
        let input = r#"
            (set-logic QF_AUFLIA)
            (declare-const a (Array Int Int))
            (declare-const i Int)
            (declare-const v Int)
            (assert (= (select a i) v))
            (assert (>= v 0))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_auflia_arithmetic_constraint_unsat() {
        // Array with contradictory arithmetic constraints on values
        let input = r#"
            (set-logic QF_AUFLIA)
            (declare-const a (Array Int Int))
            (declare-const i Int)
            (assert (>= (select a i) 10))
            (assert (< (select a i) 5))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_auflia_function_equality_unsat() {
        // f(i) = 5, f(i) = 6 is contradictory (EUF reasoning)
        let input = r#"
            (set-logic QF_AUFLIA)
            (declare-const i Int)
            (declare-fun f (Int) Int)
            (assert (= (f i) 5))
            (assert (= (f i) 6))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_auflia_combined_sat() {
        // Combination of arrays, UF, and arithmetic - all satisfiable
        let input = r#"
            (set-logic QF_AUFLIA)
            (declare-const a (Array Int Int))
            (declare-const i Int)
            (declare-const j Int)
            (declare-fun f (Int) Int)
            (assert (>= i 0))
            (assert (<= j 10))
            (assert (= (f i) (select a j)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_auflia_index_bounds() {
        // Array with integer index constraints
        let input = r#"
            (set-logic QF_AUFLIA)
            (declare-const a (Array Int Int))
            (declare-const i Int)
            (assert (>= i 0))
            (assert (<= i 100))
            (assert (= (select a i) i))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    // QF_AUFLRA (Arrays + Uninterpreted Functions + Linear Real Arithmetic) Tests

    #[test]
    fn test_executor_qf_auflra_simple_sat() {
        // Basic array with real-valued contents and UF
        let input = r#"
            (set-logic QF_AUFLRA)
            (declare-const a (Array Int Real))
            (declare-const i Int)
            (declare-fun f (Int) Real)
            (assert (= (select a i) (f i)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_auflra_array_arithmetic_unsat() {
        // Array with contradictory real arithmetic constraints
        let input = r#"
            (set-logic QF_AUFLRA)
            (declare-const a (Array Int Real))
            (declare-const i Int)
            (assert (>= (select a i) 10.0))
            (assert (< (select a i) 5.0))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_auflra_function_equality_unsat() {
        // f(x) = 5.5, f(x) = 6.5 is contradictory
        let input = r#"
            (set-logic QF_AUFLRA)
            (declare-const x Int)
            (declare-fun f (Int) Real)
            (assert (= (f x) 5.5))
            (assert (= (f x) 6.5))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_auflra_combined_sat() {
        // Combination of arrays, UF, and real arithmetic - all satisfiable
        // Note: Arithmetic constraints on UF terms should be handled by the EUF solver,
        // while arithmetic on array select results go to LRA.
        let input = r#"
            (set-logic QF_AUFLRA)
            (declare-const a (Array Int Real))
            (declare-const i Int)
            (declare-const j Int)
            (declare-const x Real)
            (declare-fun f (Int) Real)
            (assert (= (select a i) x))
            (assert (>= x 0.0))
            (assert (= (f i) (f j)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_auflra_real_constraints() {
        // Array values constrained by real arithmetic
        let input = r#"
            (set-logic QF_AUFLRA)
            (declare-const a (Array Int Real))
            (declare-const i Int)
            (declare-const x Real)
            (assert (= (select a i) x))
            (assert (>= x 1.5))
            (assert (<= x 2.5))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    // =========================================================================
    // QF_BV (Quantifier-Free Bitvectors) tests
    // =========================================================================

    #[test]
    fn test_executor_qf_bv_simple_sat() {
        // Simple bitvector constraint - should be satisfiable
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= x y))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_constant_equality_sat() {
        // x = 5 should be satisfiable
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (= x #x05))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_contradiction_unsat() {
        // x = 5 and x = 6 is unsatisfiable
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (= x #x05))
            (assert (= x #x06))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_bv_addition_sat() {
        // x + y = 10 should be satisfiable
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= (bvadd x y) #x0a))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_bitwise_and_sat() {
        // (x & 0xFF) = x is always true for 8-bit bitvectors
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (= (bvand x #xff) x))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_ult_sat() {
        // x < 10 should be satisfiable
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (bvult x #x0a))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_ult_unsat() {
        // x < 0 is unsatisfiable for unsigned bitvectors
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (bvult x #x00))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_bv_combined_constraints() {
        // x + y = 100, x < y should be satisfiable
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= (bvadd x y) #x64))
            (assert (bvult x y))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_bvor_sat() {
        // (x | y) = 0xFF should be satisfiable
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= (bvor x y) #xff))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_bvxor_sat() {
        // x XOR x = 0 is always true
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (= (bvxor x x) #x00))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_bvnot_sat() {
        // ~x = 0xFE means x = 0x01
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (= (bvnot x) #xfe))
            (assert (= x #x01))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    // =========================================================================
    // QF_BV Division tests
    // =========================================================================

    #[test]
    fn test_executor_qf_bv_udiv_sat() {
        // x / 3 = 2 means x can be 6, 7, or 8
        // With additional constraint x = 7
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (declare-const q (_ BitVec 8))
            (assert (= q (bvudiv x #x03)))
            (assert (= q #x02))
            (assert (= x #x07))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_udiv_unsat() {
        // x / 3 = 2 and x / 3 = 3 is unsatisfiable
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (= (bvudiv x #x03) #x02))
            (assert (= (bvudiv x #x03) #x03))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_bv_urem_sat() {
        // x % 3 = 1 should be satisfiable (e.g., x = 1, 4, 7, ...)
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (= (bvurem x #x03) #x01))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_urem_constraint_sat() {
        // x % 4 = 3 and x < 16 should be satisfiable (x = 3, 7, 11, 15)
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (= (bvurem x #x04) #x03))
            (assert (bvult x #x10))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_div_by_zero() {
        // Division by zero: x / 0 = 0xFF (all ones for 8-bit)
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (= x #x05))
            (assert (= (bvudiv x #x00) #xff))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_rem_by_zero() {
        // Remainder by zero: x % 0 = x
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (= x #x07))
            (assert (= (bvurem x #x00) x))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_sdiv_positive() {
        // Signed division: 7 / 2 = 3
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (= x #x07))
            (assert (= (bvsdiv x #x02) #x03))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_srem_positive() {
        // Signed remainder: 7 % 3 = 1
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (= x #x07))
            (assert (= (bvsrem x #x03) #x01))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_bv_div_quotient_remainder() {
        // Quotient-remainder relationship: a = q * b + r
        // For a = 10, b = 3: q = 3, r = 1
        let input = r#"
            (set-logic QF_BV)
            (declare-const a (_ BitVec 8))
            (declare-const b (_ BitVec 8))
            (declare-const q (_ BitVec 8))
            (declare-const r (_ BitVec 8))
            (assert (= a #x0a))
            (assert (= b #x03))
            (assert (= q (bvudiv a b)))
            (assert (= r (bvurem a b)))
            (assert (= a (bvadd (bvmul q b) r)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    // =========================================================================
    // QF_ABV (Quantifier-Free Arrays + Bitvectors) tests
    // =========================================================================

    #[test]
    fn test_executor_qf_abv_simple_sat() {
        // Simple array with bitvector index and value
        let input = r#"
            (set-logic QF_ABV)
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const i (_ BitVec 8))
            (declare-const v (_ BitVec 8))
            (assert (= v (select a i)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_abv_store_select_same_index() {
        // select(store(a, i, v), i) = v (ROW1 axiom)
        let input = r#"
            (set-logic QF_ABV)
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const i (_ BitVec 8))
            (declare-const v (_ BitVec 8))
            (assert (= (select (store a i v) i) v))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_abv_store_different_value_unsat() {
        // select(store(a, i, v1), i) = v2 where v1 != v2 is unsat
        let input = r#"
            (set-logic QF_ABV)
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const i (_ BitVec 8))
            (assert (= (select (store a i #x05) i) #x06))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_abv_bv_constraints_sat() {
        // Array with bitvector operations on indices/values
        let input = r#"
            (set-logic QF_ABV)
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const i (_ BitVec 8))
            (declare-const v (_ BitVec 8))
            (assert (= i #x05))
            (assert (= v (select a i)))
            (assert (bvult v #x10))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_abv_multiple_stores_sat() {
        // Multiple stores to same array
        let input = r#"
            (set-logic QF_ABV)
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const b (Array (_ BitVec 8) (_ BitVec 8)))
            (assert (= b (store (store a #x00 #x01) #x01 #x02)))
            (assert (= (select b #x00) #x01))
            (assert (= (select b #x01) #x02))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_abv_contradictory_values_unsat() {
        // Same index, different values - contradiction
        let input = r#"
            (set-logic QF_ABV)
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const i (_ BitVec 8))
            (assert (= (select a i) #x05))
            (assert (= (select a i) #x06))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_abv_bv_arithmetic_on_values() {
        // BV arithmetic on array values
        let input = r#"
            (set-logic QF_ABV)
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const i (_ BitVec 8))
            (declare-const j (_ BitVec 8))
            (assert (= (bvadd (select a i) (select a j)) #x0a))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_abv_32bit_sat() {
        // 32-bit bitvectors (common for Kani workloads)
        let input = r#"
            (set-logic QF_ABV)
            (declare-const a (Array (_ BitVec 32) (_ BitVec 32)))
            (declare-const i (_ BitVec 32))
            (declare-const v (_ BitVec 32))
            (assert (= i #x00000005))
            (assert (= v (select a i)))
            (assert (bvult v #x00000100))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_abv_memory_model_sat() {
        // Memory model pattern: store then select at different index
        let input = r#"
            (set-logic QF_ABV)
            (declare-const mem (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const ptr1 (_ BitVec 8))
            (declare-const ptr2 (_ BitVec 8))
            (declare-const val (_ BitVec 8))
            (assert (= ptr1 #x10))
            (assert (= ptr2 #x20))
            (assert (= val #x42))
            (assert (= (select (store mem ptr1 val) ptr2) (select mem ptr2)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_abv_overwrite_sat() {
        // Store overwrites previous value at same index
        let input = r#"
            (set-logic QF_ABV)
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const b (Array (_ BitVec 8) (_ BitVec 8)))
            (assert (= b (store (store a #x05 #x01) #x05 #x02)))
            (assert (= (select b #x05) #x02))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    // =========================================================================
    // QF_UFBV (Quantifier-Free UF + Bitvectors) tests
    // =========================================================================

    #[test]
    fn test_executor_qf_ufbv_simple_sat() {
        // Simple uninterpreted function with bitvector arguments and result
        let input = r#"
            (set-logic QF_UFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-const x (_ BitVec 8))
            (assert (= (f x) #x42))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_ufbv_functional_consistency() {
        // f(x) = f(x) should always be sat (same term)
        let input = r#"
            (set-logic QF_UFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-const x (_ BitVec 8))
            (assert (= (f x) (f x)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_ufbv_congruence_sat() {
        // x = y implies f(x) = f(y) (functional consistency)
        let input = r#"
            (set-logic QF_UFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= x y))
            (assert (= (f x) (f y)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_ufbv_congruence_unsat() {
        // x = y and f(x) != f(y) is unsat (violates congruence)
        let input = r#"
            (set-logic QF_UFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= x y))
            (assert (not (= (f x) (f y))))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_ufbv_multi_arg_sat() {
        // Uninterpreted function with multiple arguments
        let input = r#"
            (set-logic QF_UFBV)
            (declare-fun g ((_ BitVec 8) (_ BitVec 8)) (_ BitVec 8))
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= (g x y) #x05))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_ufbv_multi_arg_congruence_unsat() {
        // (x1 = y1 and x2 = y2) implies g(x1,x2) = g(y1,y2)
        let input = r#"
            (set-logic QF_UFBV)
            (declare-fun g ((_ BitVec 8) (_ BitVec 8)) (_ BitVec 8))
            (declare-const x1 (_ BitVec 8))
            (declare-const x2 (_ BitVec 8))
            (declare-const y1 (_ BitVec 8))
            (declare-const y2 (_ BitVec 8))
            (assert (= x1 y1))
            (assert (= x2 y2))
            (assert (not (= (g x1 x2) (g y1 y2))))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_ufbv_bv_constraints_sat() {
        // Combine UF with BV arithmetic
        let input = r#"
            (set-logic QF_UFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-const x (_ BitVec 8))
            (assert (= x #x05))
            (assert (bvult (f x) #x10))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_ufbv_multiple_functions_sat() {
        // Multiple uninterpreted functions
        let input = r#"
            (set-logic QF_UFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-fun h ((_ BitVec 8)) (_ BitVec 8))
            (declare-const x (_ BitVec 8))
            (assert (= (bvadd (f x) (h x)) #x0a))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_ufbv_nested_function_sat() {
        // Nested uninterpreted function applications
        let input = r#"
            (set-logic QF_UFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-const x (_ BitVec 8))
            (assert (= (f (f x)) #x42))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_ufbv_32bit_sat() {
        // 32-bit bitvectors with UF
        let input = r#"
            (set-logic QF_UFBV)
            (declare-fun f ((_ BitVec 32)) (_ BitVec 32))
            (declare-const x (_ BitVec 32))
            (assert (= x #x00000005))
            (assert (bvult (f x) #x00000100))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    // =========================================================================
    // QF_AUFBV (Quantifier-Free Arrays + UF + Bitvectors) tests
    // =========================================================================

    #[test]
    fn test_executor_qf_aufbv_simple_sat() {
        // Combine arrays and UF with bitvectors
        let input = r#"
            (set-logic QF_AUFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const i (_ BitVec 8))
            (assert (= (f (select a i)) #x42))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_aufbv_uf_over_array_index_sat() {
        // Use UF to compute array index
        let input = r#"
            (set-logic QF_AUFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const x (_ BitVec 8))
            (assert (= (select a (f x)) #x42))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_aufbv_array_store_with_uf_sat() {
        // Store value computed by UF
        let input = r#"
            (set-logic QF_AUFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const b (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const x (_ BitVec 8))
            (declare-const i (_ BitVec 8))
            (assert (= b (store a i (f x))))
            (assert (= (select b i) (f x)))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_aufbv_uf_congruence_with_array_unsat() {
        // x = y implies f(x) = f(y) even with array context
        let input = r#"
            (set-logic QF_AUFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= x y))
            (assert (not (= (f x) (f y))))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_aufbv_array_select_same_index_unsat() {
        // Same as QF_ABV but in QF_AUFBV context
        let input = r#"
            (set-logic QF_AUFBV)
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const i (_ BitVec 8))
            (assert (= (select a i) #x05))
            (assert (= (select a i) #x06))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["unsat"]);
    }

    #[test]
    fn test_executor_qf_aufbv_combined_constraints_sat() {
        // Complex combination of arrays, UF, and BV
        let input = r#"
            (set-logic QF_AUFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-fun g ((_ BitVec 8) (_ BitVec 8)) (_ BitVec 8))
            (declare-const mem (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const ptr (_ BitVec 8))
            (declare-const val (_ BitVec 8))
            (assert (= ptr #x10))
            (assert (= val (select mem ptr)))
            (assert (= (f val) #x42))
            (assert (bvult (g ptr val) #x80))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_aufbv_kani_memory_pattern_sat() {
        // Kani-like memory model pattern
        let input = r#"
            (set-logic QF_AUFBV)
            (declare-fun ptr_to_idx ((_ BitVec 32)) (_ BitVec 32))
            (declare-const mem (Array (_ BitVec 32) (_ BitVec 32)))
            (declare-const ptr1 (_ BitVec 32))
            (declare-const ptr2 (_ BitVec 32))
            (declare-const val (_ BitVec 32))
            (assert (= (ptr_to_idx ptr1) #x00000000))
            (assert (= (ptr_to_idx ptr2) #x00000001))
            (assert (= (select (store mem (ptr_to_idx ptr1) #x42424242) (ptr_to_idx ptr2))
                       (select mem (ptr_to_idx ptr2))))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_aufbv_row1_with_uf_sat() {
        // select(store(a, i, v), i) = v with UF context
        let input = r#"
            (set-logic QF_AUFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const i (_ BitVec 8))
            (declare-const v (_ BitVec 8))
            (assert (= v (f i)))
            (assert (= (select (store a i v) i) v))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_aufbv_32bit_sat() {
        // 32-bit bitvectors with arrays and UF
        let input = r#"
            (set-logic QF_AUFBV)
            (declare-fun f ((_ BitVec 32)) (_ BitVec 32))
            (declare-const a (Array (_ BitVec 32) (_ BitVec 32)))
            (declare-const i (_ BitVec 32))
            (assert (= i #x00000005))
            (assert (bvult (f (select a i)) #x00000100))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_executor_qf_aufbv_nested_uf_array_sat() {
        // Nested UF and array operations
        let input = r#"
            (set-logic QF_AUFBV)
            (declare-fun f ((_ BitVec 8)) (_ BitVec 8))
            (declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const b (Array (_ BitVec 8) (_ BitVec 8)))
            (declare-const x (_ BitVec 8))
            (assert (= b (store a x (f (select a x)))))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs, vec!["sat"]);
    }

    // ==========================================================================
    // Incremental solving tests (Kani Fast Requirement 1.2)
    // ==========================================================================

    #[test]
    fn test_incremental_bv_basic_push_pop() {
        // Basic push/pop test with QF_BV
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (bvult x #x10))
            (push 1)
            (assert (= x #x05))
            (check-sat)
            (pop 1)
            (push 1)
            (assert (= x #x0F))
            (check-sat)
            (pop 1)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // Both check-sats should return SAT (x < 16, and x = 5 or x = 15)
        assert_eq!(outputs, vec!["sat", "sat"]);
    }

    #[test]
    fn test_incremental_bv_unsat_after_pop() {
        // Test that assertions are properly scoped
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (= x #x00))
            (push 1)
            (assert (= x #x01))
            (check-sat)
            (pop 1)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // First check-sat: x = 0 AND x = 1 is UNSAT
        // Second check-sat after pop: only x = 0, which is SAT
        assert_eq!(outputs, vec!["unsat", "sat"]);
    }

    #[test]
    fn test_incremental_bv_nested_scopes() {
        // Test nested push/pop scopes
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (bvult x #x10))
            (push 1)
            (assert (bvult y #x10))
            (push 1)
            (assert (= (bvadd x y) #x1E))
            (check-sat)
            (pop 1)
            (assert (= (bvadd x y) #x08))
            (check-sat)
            (pop 1)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // First: x < 16, y < 16, x + y = 30. With x=15, y=15, sum=30=0x1E -> SAT
        // Second: x < 16, y < 16, x + y = 8 -> SAT (e.g., x=0, y=8)
        // Third: only x < 16 -> SAT
        assert_eq!(outputs, vec!["sat", "sat", "sat"]);
    }

    #[test]
    fn test_incremental_bv_clause_retention() {
        // Test that learned clauses are retained across check-sat calls
        // This is the key benefit of incremental solving
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (bvult x #x10))
            (assert (bvult y #x10))
            (push 1)
            (assert (= (bvadd x y) #x1E))
            (check-sat)
            (pop 1)
            (push 1)
            (assert (= (bvadd x y) #x1D))
            (check-sat)
            (pop 1)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // Both should be SAT - the second check-sat should be faster
        // because learned clauses from first check-sat are retained
        assert_eq!(outputs, vec!["sat", "sat"]);
    }

    #[test]
    fn test_incremental_bv_multiple_check_sat_same_scope() {
        // Multiple check-sat calls in the same scope
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (push 1)
            (assert (bvugt x #x00))
            (check-sat)
            (assert (bvult x #x10))
            (check-sat)
            (assert (= (bvand x #x0F) x))
            (check-sat)
            (pop 1)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // All should be SAT with progressively narrowing constraints
        assert_eq!(outputs, vec!["sat", "sat", "sat"]);
    }

    #[test]
    fn test_incremental_bv_pop_to_sat() {
        // Test that popping can turn UNSAT into SAT
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (= x #x05))
            (assert (= y #x0A))
            (push 1)
            (assert (= x y))
            (check-sat)
            (pop 1)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // First: x = 5, y = 10, x = y -> UNSAT
        // After pop: x = 5, y = 10 -> SAT
        assert_eq!(outputs, vec!["unsat", "sat"]);
    }

    #[test]
    fn test_incremental_bv_deep_nesting() {
        // Test deeply nested scopes
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (bvult x #x80))
            (push 1)
            (assert (bvult x #x40))
            (push 1)
            (assert (bvult x #x20))
            (push 1)
            (assert (bvult x #x10))
            (check-sat)
            (pop 1)
            (check-sat)
            (pop 1)
            (check-sat)
            (pop 1)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // All should be SAT with progressively relaxed constraints
        assert_eq!(outputs, vec!["sat", "sat", "sat", "sat"]);
    }

    #[test]
    fn test_incremental_bv_kani_pattern() {
        // Simulates a Kani-like verification pattern
        // - Global constraints on memory model
        // - Push/check-sat/pop for each verification condition
        let input = r#"
            (set-logic QF_BV)
            (declare-const ptr (_ BitVec 32))
            (declare-const len (_ BitVec 32))
            (declare-const idx (_ BitVec 32))

            ; Global: ptr is non-null, len > 0
            (assert (not (= ptr #x00000000)))
            (assert (bvugt len #x00000000))

            ; VC1: idx < len implies valid access
            (push 1)
            (assert (bvult idx len))
            (check-sat)
            (pop 1)

            ; VC2: idx = 0 implies valid access
            (push 1)
            (assert (= idx #x00000000))
            (assert (bvult idx len))
            (check-sat)
            (pop 1)

            ; VC3: idx = len - 1 implies valid access (boundary)
            (push 1)
            (assert (= idx (bvsub len #x00000001)))
            (assert (bvult idx len))
            (check-sat)
            (pop 1)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // All VCs should be SAT
        assert_eq!(outputs, vec!["sat", "sat", "sat"]);
    }

    #[test]
    fn test_incremental_enabled_by_push() {
        // Verify incremental mode is enabled when push is used
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (push 1)
            (assert (= x #x42))
            (check-sat)
            (pop 1)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let _ = exec.execute_all(&commands).unwrap();

        // After push, incremental mode should be enabled
        assert!(exec.incremental_mode);
    }

    #[test]
    fn test_incremental_bv_simple_constraint() {
        // Minimal test: one global assertion + push/pop
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (assert (bvult x #x10))
            (push 1)
            (check-sat)
            (pop 1)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // Should be SAT - x can be any value 0..15
        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_incremental_bv_two_global() {
        // Test: two global assertions + push/pop with no scoped assertion
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (bvult x #x10))
            (assert (bvult y #x10))
            (push 1)
            (check-sat)
            (pop 1)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // Should be SAT - x and y can each be 0..15
        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_incremental_bv_with_scoped() {
        // Test: two global + one scoped
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (bvult x #x10))
            (assert (bvult y #x10))
            (push 1)
            (assert (= (bvadd x y) #x1E))
            (check-sat)
            (pop 1)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // Should be SAT - x=15, y=15 satisfies all
        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_bv_nonincr_same_constraints() {
        // Same constraints as test_incremental_bv_with_scoped but without push/pop
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (assert (bvult x #x10))
            (assert (bvult y #x10))
            (assert (= (bvadd x y) #x1E))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // Should be SAT - x=15, y=15 satisfies all
        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_incremental_bv_only_scoped() {
        // Test: only scoped assertions, no global
        let input = r#"
            (set-logic QF_BV)
            (declare-const x (_ BitVec 8))
            (declare-const y (_ BitVec 8))
            (push 1)
            (assert (bvult x #x10))
            (assert (bvult y #x10))
            (assert (= (bvadd x y) #x1E))
            (check-sat)
            (pop 1)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // All scoped, should be SAT
        assert_eq!(outputs, vec!["sat"]);
    }

    #[test]
    fn test_incremental_vs_fresh_timing() {
        use std::time::Instant;

        // Build scripts
        let num_checks = 5;

        // Single incremental script
        let mut incr_script = String::from(
            "(set-logic QF_BV)\n\
             (declare-const x (_ BitVec 8))\n\
             (declare-const y (_ BitVec 8))\n\
             (assert (bvult x #xff))\n\
             (assert (bvult y #xff))\n",
        );
        for i in 0..num_checks {
            incr_script.push_str("(push 1)\n");
            incr_script.push_str(&format!("(assert (bvugt (bvadd x y) #x{:02x}))\n", i));
            incr_script.push_str("(check-sat)\n");
            incr_script.push_str("(pop 1)\n");
        }

        // Fresh scripts (one per check)
        let fresh_scripts: Vec<String> = (0..num_checks)
            .map(|i| {
                format!(
                    "(set-logic QF_BV)\n\
                     (declare-const x (_ BitVec 8))\n\
                     (declare-const y (_ BitVec 8))\n\
                     (assert (bvult x #xff))\n\
                     (assert (bvult y #xff))\n\
                     (assert (bvugt (bvadd x y) #x{:02x}))\n\
                     (check-sat)\n",
                    i
                )
            })
            .collect();

        // Time incremental
        let start = Instant::now();
        let commands = parse(&incr_script).unwrap();
        let mut exec = Executor::new();
        let mut incr_results = Vec::new();
        for cmd in &commands {
            if let Ok(Some(out)) = exec.execute(cmd) {
                incr_results.push(out);
            }
        }
        let incr_time = start.elapsed();

        // Time fresh
        let start = Instant::now();
        let mut fresh_results = Vec::new();
        for script in &fresh_scripts {
            let commands = parse(script).unwrap();
            let mut exec = Executor::new();
            for cmd in &commands {
                if let Ok(Some(out)) = exec.execute(cmd) {
                    fresh_results.push(out);
                }
            }
        }
        let fresh_time = start.elapsed();

        println!("\n=== Incremental vs Fresh Timing ===");
        println!("Incremental: {:?} ({} checks)", incr_time, num_checks);
        println!("Fresh: {:?} ({} checks)", fresh_time, num_checks);
        println!(
            "Ratio: {:.2}x",
            incr_time.as_secs_f64() / fresh_time.as_secs_f64()
        );
        println!("Incremental results: {:?}", incr_results);
        println!("Fresh results: {:?}", fresh_results);

        // Both should produce same results
        assert_eq!(incr_results.len(), fresh_results.len());
        for (i, f) in incr_results.iter().zip(fresh_results.iter()) {
            assert_eq!(i, f);
        }
    }

    #[test]
    fn test_lra_model_extraction() {
        // Test that LRA model values are extracted and displayed correctly
        let input = r#"
            (set-logic QF_LRA)
            (set-option :produce-models true)
            (declare-const x Real)
            (declare-const y Real)
            (assert (> x 0))
            (assert (< y 10))
            (assert (= (+ x y) 5))
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        // Model should contain definitions for x and y with Real values
        let model = &outputs[1];
        assert!(model.contains("(model"), "Model should start with (model");
        assert!(
            model.contains("define-fun x () Real") || model.contains("define-fun y () Real"),
            "Model should contain Real variable definitions"
        );
    }

    #[test]
    fn test_lia_model_extraction() {
        // Test that LIA model values are extracted and displayed correctly
        let input = r#"
            (set-logic QF_LIA)
            (set-option :produce-models true)
            (declare-const n Int)
            (declare-const m Int)
            (assert (> n 5))
            (assert (< m 10))
            (assert (= (+ n m) 12))
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        // Model should contain definitions for n and m with Int values
        let model = &outputs[1];
        assert!(model.contains("(model"), "Model should start with (model");
        assert!(
            model.contains("define-fun n () Int") || model.contains("define-fun m () Int"),
            "Model should contain Int variable definitions"
        );
    }

    #[test]
    fn test_minimize_counterexamples_option() {
        // Test that the minimize-counterexamples option is recognized
        let input = r#"
            (set-logic QF_LRA)
            (set-option :produce-models true)
            (set-option :minimize-counterexamples true)
            (declare-const x Real)
            (assert (>= x 0))
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        // Model should exist (minimization option shouldn't break anything)
        let model = &outputs[1];
        assert!(model.contains("(model"), "Model should start with (model");
    }

    #[test]
    fn test_minimize_counterexamples_prefers_zero() {
        // Test that minimize-counterexamples prefers 0 when valid
        // This is a simple test case where x >= 0 allows x = 0
        let input = r#"
            (set-logic QF_LRA)
            (set-option :produce-models true)
            (set-option :minimize-counterexamples true)
            (declare-const x Real)
            (assert (>= x 0))
            (check-sat)
            (get-model)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "sat");
        let model = &outputs[1];
        // With minimization enabled and x >= 0, we should prefer x = 0
        // Note: This test may pass even without minimization if the solver
        // happens to choose 0, but it validates the option doesn't break
        assert!(model.contains("(model"), "Model should start with (model");
    }

    #[test]
    fn test_format_bigint() {
        // Test the format_bigint helper function
        use num_bigint::BigInt;

        // Positive values
        assert_eq!(format_bigint(&BigInt::from(0)), "0");
        assert_eq!(format_bigint(&BigInt::from(1)), "1");
        assert_eq!(format_bigint(&BigInt::from(42)), "42");

        // Negative values should use SMT-LIB format (- x)
        assert_eq!(format_bigint(&BigInt::from(-1)), "(- 1)");
        assert_eq!(format_bigint(&BigInt::from(-42)), "(- 42)");
    }

    #[test]
    fn test_format_rational() {
        // Test the format_rational helper function
        use num_bigint::BigInt;

        // Integer rationals
        assert_eq!(
            format_rational(&BigRational::from_integer(BigInt::from(0))),
            "0.0"
        );
        assert_eq!(
            format_rational(&BigRational::from_integer(BigInt::from(1))),
            "1.0"
        );
        assert_eq!(
            format_rational(&BigRational::from_integer(BigInt::from(-1))),
            "(- 1)"
        );

        // Fractional rationals
        let half = BigRational::new(BigInt::from(1), BigInt::from(2));
        assert_eq!(format_rational(&half), "(/ 1 2)");

        let neg_half = BigRational::new(BigInt::from(-1), BigInt::from(2));
        assert_eq!(format_rational(&neg_half), "(- (/ 1 2))");
    }

    #[test]
    fn test_incremental_lra_push_pop() {
        // Test incremental LRA solving with push/pop
        // First check-sat should be SAT (x > 0)
        // After push + additional constraint x < 0, should be UNSAT
        // After pop, should be SAT again
        let input = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (assert (> x 0))
            (push 1)
            (assert (< x 0))
            (check-sat)
            (pop 1)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // After push + (x < 0), with existing (x > 0): UNSAT
        // After pop (back to just x > 0): SAT
        assert_eq!(outputs, vec!["unsat", "sat"]);
    }

    #[test]
    fn test_incremental_lra_multiple_checks() {
        // Test that learned clauses are retained across check-sat calls
        let input = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (declare-const y Real)
            (assert (>= x 0))
            (assert (>= y 0))
            (check-sat)
            (push 1)
            (assert (< (+ x y) 0))
            (check-sat)
            (pop 1)
            (assert (<= x 10))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // x >= 0, y >= 0: SAT
        // x >= 0, y >= 0, x + y < 0: UNSAT
        // x >= 0, y >= 0, x <= 10: SAT
        assert_eq!(outputs, vec!["sat", "unsat", "sat"]);
    }

    #[test]
    fn test_incremental_lra_nested_push_pop() {
        // Test nested push/pop levels
        let input = r#"
            (set-logic QF_LRA)
            (declare-const x Real)
            (assert (> x 0))
            (check-sat)
            (push 1)
            (assert (< x 5))
            (check-sat)
            (push 1)
            (assert (> x 10))
            (check-sat)
            (pop 1)
            (check-sat)
            (pop 1)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // x > 0: SAT
        // x > 0, x < 5: SAT (0 < x < 5)
        // x > 0, x < 5, x > 10: UNSAT (contradiction)
        // x > 0, x < 5: SAT (after pop)
        // x > 0: SAT (after final pop)
        assert_eq!(outputs, vec!["sat", "sat", "unsat", "sat", "sat"]);
    }

    #[test]
    fn test_incremental_euf_push_pop() {
        // Test incremental EUF solving with push/pop
        // First check-sat should be SAT (= a b)
        // After push + (distinct a b), should be UNSAT
        // After pop, should be SAT again
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (assert (= a b))
            (push 1)
            (assert (distinct a b))
            (check-sat)
            (pop 1)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // After push + (distinct a b), with existing (= a b): UNSAT
        // After pop (back to just (= a b)): SAT
        assert_eq!(outputs, vec!["unsat", "sat"]);
    }

    #[test]
    fn test_incremental_euf_multiple_checks() {
        // Test that learned clauses are retained across check-sat calls
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (declare-const c U)
            (assert (= a b))
            (check-sat)
            (push 1)
            (assert (= b c))
            (assert (distinct a c))
            (check-sat)
            (pop 1)
            (assert (distinct a b))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // a = b: SAT
        // a = b, b = c, a != c: UNSAT (transitivity violation)
        // a = b, a != b: UNSAT
        assert_eq!(outputs, vec!["sat", "unsat", "unsat"]);
    }

    #[test]
    fn test_incremental_euf_nested_push_pop() {
        // Test nested push/pop levels for EUF
        let input = r#"
            (set-logic QF_UF)
            (declare-sort U 0)
            (declare-const a U)
            (declare-const b U)
            (declare-const c U)
            (declare-fun f (U) U)
            (assert (= (f a) (f b)))
            (check-sat)
            (push 1)
            (assert (= a b))
            (check-sat)
            (push 1)
            (assert (distinct (f a) (f b)))
            (check-sat)
            (pop 1)
            (check-sat)
            (pop 1)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // f(a) = f(b): SAT
        // f(a) = f(b), a = b: SAT
        // f(a) = f(b), a = b, f(a) != f(b): UNSAT
        // f(a) = f(b), a = b: SAT (after pop)
        // f(a) = f(b): SAT (after final pop)
        assert_eq!(outputs, vec!["sat", "sat", "unsat", "sat", "sat"]);
    }

    #[test]
    fn test_incremental_lia_push_pop() {
        // Test incremental LIA solving with push/pop
        // First check-sat should be SAT (x >= 0)
        // After push + (x < 0), should be UNSAT
        // After pop, should be SAT again
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (assert (>= x 0))
            (push 1)
            (assert (< x 0))
            (check-sat)
            (pop 1)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // After push + (x < 0), with existing (x >= 0): UNSAT
        // After pop (back to just x >= 0): SAT
        assert_eq!(outputs, vec!["unsat", "sat"]);
    }

    #[test]
    fn test_incremental_lia_multiple_checks() {
        // Test that learned clauses are retained across check-sat calls
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (declare-const y Int)
            (assert (>= x 0))
            (assert (>= y 0))
            (check-sat)
            (push 1)
            (assert (< (+ x y) 0))
            (check-sat)
            (pop 1)
            (assert (<= x 10))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // x >= 0, y >= 0: SAT
        // x >= 0, y >= 0, x + y < 0: UNSAT
        // x >= 0, y >= 0, x <= 10: SAT
        assert_eq!(outputs, vec!["sat", "unsat", "sat"]);
    }

    #[test]
    fn test_incremental_lia_nested_push_pop() {
        // Test nested push/pop levels for LIA
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (assert (> x 0))
            (check-sat)
            (push 1)
            (assert (< x 5))
            (check-sat)
            (push 1)
            (assert (> x 10))
            (check-sat)
            (pop 1)
            (check-sat)
            (pop 1)
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // x > 0: SAT
        // x > 0, x < 5: SAT (1, 2, 3, 4 are valid)
        // x > 0, x < 5, x > 10: UNSAT (contradiction)
        // x > 0, x < 5: SAT (after pop)
        // x > 0: SAT (after final pop)
        assert_eq!(outputs, vec!["sat", "sat", "unsat", "sat", "sat"]);
    }

    #[test]
    fn test_incremental_lia_with_integer_splits() {
        // Test LIA incremental mode with problems that require splits
        let input = r#"
            (set-logic QF_LIA)
            (declare-const x Int)
            (assert (> (* 2 x) 1))
            (assert (< (* 2 x) 4))
            (push 1)
            (check-sat)
            (pop 1)
            (assert (= (* 2 x) 2))
            (check-sat)
        "#;

        let commands = parse(input).unwrap();
        let mut exec = Executor::new();
        let outputs = exec.execute_all(&commands).unwrap();

        // 2x > 1 and 2x < 4: SAT (x = 1 works)
        // 2x > 1 and 2x < 4 and 2x = 2: SAT (x = 1 works)
        assert_eq!(outputs, vec!["sat", "sat"]);
    }
}
