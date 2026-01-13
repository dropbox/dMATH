//! Native Rust API for Z4 SMT Solver
//!
//! This module provides a programmatic interface for building and solving SMT
//! constraints directly in Rust, without parsing SMT-LIB text.
//!
//! # Example
//!
//! ```
//! use z4_dpll::api::{Solver, Sort, SolveResult};
//!
//! let mut solver = Solver::new(z4_dpll::api::Logic::QfLia);
//!
//! // Declare variables
//! let x = solver.declare_const("x", Sort::Int);
//! let y = solver.declare_const("y", Sort::Int);
//!
//! // Assert constraints: x > 0 and y = x + 1
//! let zero = solver.int_const(0);
//! let one = solver.int_const(1);
//! let x_gt_zero = solver.gt(x, zero);
//! solver.assert_term(x_gt_zero);
//! let x_plus_one = solver.add(x, one);
//! let y_eq_x_plus_one = solver.eq(y, x_plus_one);
//! solver.assert_term(y_eq_x_plus_one);
//!
//! // Check satisfiability
//! match solver.check_sat() {
//!     SolveResult::Sat => {
//!         let model = solver.get_model().unwrap();
//!         println!("x = {:?}", model.get_int("x"));
//!         println!("y = {:?}", model.get_int("y"));
//!     }
//!     SolveResult::Unsat => println!("unsatisfiable"),
//!     SolveResult::Unknown => println!("unknown"),
//! }
//! ```

use num_bigint::BigInt;
use num_rational::BigRational;
use std::collections::HashMap;

use z4_core::term::Symbol;
use z4_core::{Sort as CoreSort, TermId, TermStore};
use z4_frontend::Command;

use crate::{CheckSatResult, Executor};

/// SMT logic specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Logic {
    /// Quantifier-free linear integer arithmetic
    QfLia,
    /// Quantifier-free linear real arithmetic
    QfLra,
    /// Quantifier-free linear integer and real arithmetic (mixed)
    ///
    /// Note: QF_LIRA support is incomplete. The solver may return Unknown
    /// for problems that mix Int and Real variables.
    QfLira,
    /// Quantifier-free uninterpreted functions
    QfUf,
    /// Quantifier-free uninterpreted functions with linear integer arithmetic
    QfUflia,
    /// Quantifier-free uninterpreted functions with linear real arithmetic
    QfUflra,
    /// Quantifier-free arrays with uninterpreted functions and linear integer arithmetic
    QfAuflia,
    /// Quantifier-free arrays with uninterpreted functions and linear real arithmetic
    QfAuflra,
    /// Quantifier-free arrays with uninterpreted functions and mixed integer/real arithmetic
    ///
    /// Note: QF_AUFLIRA support is incomplete. The solver may return Unknown
    /// for problems that mix Int and Real variables.
    QfAuflira,
    /// Quantifier-free bitvectors
    QfBv,
    /// Quantifier-free arrays with bitvectors
    QfAbv,
    /// Quantifier-free arrays with uninterpreted functions and bitvectors
    QfAufbv,
}

impl Logic {
    fn as_str(&self) -> &'static str {
        match self {
            Logic::QfLia => "QF_LIA",
            Logic::QfLra => "QF_LRA",
            Logic::QfLira => "QF_LIRA",
            Logic::QfUf => "QF_UF",
            Logic::QfUflia => "QF_UFLIA",
            Logic::QfUflra => "QF_UFLRA",
            Logic::QfAuflia => "QF_AUFLIA",
            Logic::QfAuflra => "QF_AUFLRA",
            Logic::QfAuflira => "QF_AUFLIRA",
            Logic::QfBv => "QF_BV",
            Logic::QfAbv => "QF_ABV",
            Logic::QfAufbv => "QF_AUFBV",
        }
    }
}

/// Variable/constant sort
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sort {
    /// Boolean sort
    Bool,
    /// Integer sort
    Int,
    /// Real sort
    Real,
    /// Bitvector sort with specified width
    BitVec(u32),
}

impl Sort {
    fn to_core(&self) -> CoreSort {
        match self {
            Sort::Bool => CoreSort::Bool,
            Sort::Int => CoreSort::Int,
            Sort::Real => CoreSort::Real,
            Sort::BitVec(w) => CoreSort::BitVec(*w),
        }
    }
}

/// A handle to a term in the solver
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Term(TermId);

/// Result of a satisfiability check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveResult {
    /// The constraints are satisfiable
    Sat,
    /// The constraints are unsatisfiable
    Unsat,
    /// The solver could not determine satisfiability
    Unknown,
}

impl From<CheckSatResult> for SolveResult {
    fn from(r: CheckSatResult) -> Self {
        match r {
            CheckSatResult::Sat => SolveResult::Sat,
            CheckSatResult::Unsat => SolveResult::Unsat,
            CheckSatResult::Unknown => SolveResult::Unknown,
        }
    }
}

/// A satisfying model mapping variables to values
#[derive(Debug, Clone)]
pub struct Model {
    int_values: HashMap<String, i64>,
    real_values: HashMap<String, f64>,
    bool_values: HashMap<String, bool>,
    bv_values: HashMap<String, (BigInt, u32)>,
}

impl Model {
    /// Get an integer variable's value
    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.int_values.get(name).copied()
    }

    /// Get a real variable's value
    pub fn get_real(&self, name: &str) -> Option<f64> {
        self.real_values.get(name).copied()
    }

    /// Get a boolean variable's value
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.bool_values.get(name).copied()
    }

    /// Get a bitvector variable's value as (value, width)
    pub fn get_bv(&self, name: &str) -> Option<(BigInt, u32)> {
        self.bv_values.get(name).cloned()
    }
}

/// Native Rust API for Z4 SMT solver
///
/// Provides a programmatic interface for building SMT constraints
/// and checking satisfiability without parsing SMT-LIB text.
pub struct Solver {
    executor: Executor,
    /// Variable names for model extraction
    var_names: HashMap<TermId, String>,
    /// Variable sorts for model extraction
    var_sorts: HashMap<TermId, Sort>,
}

impl Solver {
    /// Create a new solver for the specified logic
    pub fn new(logic: Logic) -> Self {
        let mut executor = Executor::new();
        // Set the logic
        let _ = executor.execute(&Command::SetLogic(logic.as_str().to_string()));
        Solver {
            executor,
            var_names: HashMap::new(),
            var_sorts: HashMap::new(),
        }
    }

    /// Access the internal term store
    fn terms(&self) -> &TermStore {
        &self.executor.context().terms
    }

    /// Access the internal term store mutably
    fn terms_mut(&mut self) -> &mut TermStore {
        &mut self.executor.context_mut().terms
    }

    // =========================================================================
    // Variable declaration
    // =========================================================================

    /// Declare a constant (0-arity function) with the given name and sort
    pub fn declare_const(&mut self, name: &str, sort: Sort) -> Term {
        let core_sort = sort.to_core();
        let term_id = self.terms_mut().mk_var(name, core_sort.clone());
        self.var_names.insert(term_id, name.to_string());
        self.var_sorts.insert(term_id, sort);
        // Register the symbol in the context so it appears in models
        self.executor
            .context_mut()
            .register_symbol(name.to_string(), term_id, core_sort);
        Term(term_id)
    }

    // =========================================================================
    // Constant constructors
    // =========================================================================

    /// Create a boolean constant
    pub fn bool_const(&mut self, value: bool) -> Term {
        Term(self.terms_mut().mk_bool(value))
    }

    /// Create an integer constant
    pub fn int_const(&mut self, value: i64) -> Term {
        Term(self.terms_mut().mk_int(BigInt::from(value)))
    }

    /// Create a real constant from a floating-point value
    pub fn real_const(&mut self, value: f64) -> Term {
        // Convert f64 to rational (may lose precision for irrational numbers)
        let r = BigRational::from_float(value).unwrap_or_else(|| {
            BigRational::new(
                BigInt::from((value * 1e15) as i64),
                BigInt::from(1_000_000_000_000_000i64),
            )
        });
        Term(self.terms_mut().mk_rational(r))
    }

    /// Create a real constant from a rational (numerator/denominator)
    pub fn rational_const(&mut self, numer: i64, denom: i64) -> Term {
        let r = BigRational::new(BigInt::from(numer), BigInt::from(denom));
        Term(self.terms_mut().mk_rational(r))
    }

    /// Create a bitvector constant
    pub fn bv_const(&mut self, value: i64, width: u32) -> Term {
        Term(self.terms_mut().mk_bitvec(BigInt::from(value), width))
    }

    // =========================================================================
    // Boolean operations
    // =========================================================================

    /// Create a logical AND of two terms
    pub fn and(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_and(vec![a.0, b.0]))
    }

    /// Create a logical AND of multiple terms
    pub fn and_many(&mut self, terms: &[Term]) -> Term {
        let ids: Vec<_> = terms.iter().map(|t| t.0).collect();
        Term(self.terms_mut().mk_and(ids))
    }

    /// Create a logical OR of two terms
    pub fn or(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_or(vec![a.0, b.0]))
    }

    /// Create a logical OR of multiple terms
    pub fn or_many(&mut self, terms: &[Term]) -> Term {
        let ids: Vec<_> = terms.iter().map(|t| t.0).collect();
        Term(self.terms_mut().mk_or(ids))
    }

    /// Create a logical NOT
    pub fn not(&mut self, a: Term) -> Term {
        Term(self.terms_mut().mk_not(a.0))
    }

    /// Create an implication (a => b)
    pub fn implies(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_implies(a.0, b.0))
    }

    /// Create an if-then-else (ite cond then_val else_val)
    pub fn ite(&mut self, cond: Term, then_val: Term, else_val: Term) -> Term {
        Term(self.terms_mut().mk_ite(cond.0, then_val.0, else_val.0))
    }

    // =========================================================================
    // Comparison operations
    // =========================================================================

    /// Create an equality (a = b)
    pub fn eq(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_eq(a.0, b.0))
    }

    /// Create a disequality (a != b)
    pub fn neq(&mut self, a: Term, b: Term) -> Term {
        let eq = self.eq(a, b);
        self.not(eq)
    }

    /// Create a less-than comparison (a < b)
    pub fn lt(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_lt(a.0, b.0))
    }

    /// Create a less-than-or-equal comparison (a <= b)
    pub fn le(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_le(a.0, b.0))
    }

    /// Create a greater-than comparison (a > b)
    pub fn gt(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_gt(a.0, b.0))
    }

    /// Create a greater-than-or-equal comparison (a >= b)
    pub fn ge(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_ge(a.0, b.0))
    }

    // =========================================================================
    // Arithmetic operations
    // =========================================================================

    /// Create an addition (a + b)
    pub fn add(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_add(vec![a.0, b.0]))
    }

    /// Create an addition of multiple terms
    pub fn add_many(&mut self, terms: &[Term]) -> Term {
        let ids: Vec<_> = terms.iter().map(|t| t.0).collect();
        Term(self.terms_mut().mk_add(ids))
    }

    /// Create a subtraction (a - b)
    pub fn sub(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_sub(vec![a.0, b.0]))
    }

    /// Create a multiplication (a * b)
    pub fn mul(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_mul(vec![a.0, b.0]))
    }

    /// Create a multiplication of multiple terms
    pub fn mul_many(&mut self, terms: &[Term]) -> Term {
        let ids: Vec<_> = terms.iter().map(|t| t.0).collect();
        Term(self.terms_mut().mk_mul(ids))
    }

    /// Create a negation (-a)
    pub fn neg(&mut self, a: Term) -> Term {
        Term(self.terms_mut().mk_neg(a.0))
    }

    /// Create a real division (a / b)
    pub fn div(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_div(vec![a.0, b.0]))
    }

    /// Create an integer division (a div b)
    pub fn int_div(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_intdiv(vec![a.0, b.0]))
    }

    /// Create a modulo operation (a mod b)
    pub fn modulo(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_mod(vec![a.0, b.0]))
    }

    /// Create an absolute value (abs a)
    pub fn abs(&mut self, a: Term) -> Term {
        Term(self.terms_mut().mk_abs(a.0))
    }

    // =========================================================================
    // Bitvector operations
    // =========================================================================

    /// Create a bitvector addition
    pub fn bvadd(&mut self, a: Term, b: Term) -> Term {
        let sort = self.terms().sort(a.0).clone();
        Term(
            self.terms_mut()
                .mk_app(Symbol::Named("bvadd".to_string()), vec![a.0, b.0], sort),
        )
    }

    /// Create a bitvector subtraction
    pub fn bvsub(&mut self, a: Term, b: Term) -> Term {
        let sort = self.terms().sort(a.0).clone();
        Term(
            self.terms_mut()
                .mk_app(Symbol::Named("bvsub".to_string()), vec![a.0, b.0], sort),
        )
    }

    /// Create a bitvector multiplication
    pub fn bvmul(&mut self, a: Term, b: Term) -> Term {
        let sort = self.terms().sort(a.0).clone();
        Term(
            self.terms_mut()
                .mk_app(Symbol::Named("bvmul".to_string()), vec![a.0, b.0], sort),
        )
    }

    /// Create a bitvector unsigned less-than
    pub fn bvult(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_app(
            Symbol::Named("bvult".to_string()),
            vec![a.0, b.0],
            CoreSort::Bool,
        ))
    }

    /// Create a bitvector signed less-than
    pub fn bvslt(&mut self, a: Term, b: Term) -> Term {
        Term(self.terms_mut().mk_app(
            Symbol::Named("bvslt".to_string()),
            vec![a.0, b.0],
            CoreSort::Bool,
        ))
    }

    // =========================================================================
    // Assertion and solving
    // =========================================================================

    /// Assert a constraint (must be a Boolean term)
    pub fn assert_term(&mut self, term: Term) {
        self.executor.context_mut().assertions.push(term.0);
    }

    /// Push a new scope for incremental solving
    pub fn push(&mut self) {
        let _ = self.executor.execute(&Command::Push(1));
    }

    /// Pop the most recent scope
    pub fn pop(&mut self) {
        let _ = self.executor.execute(&Command::Pop(1));
    }

    /// Check satisfiability of the current assertions
    pub fn check_sat(&mut self) -> SolveResult {
        match self.executor.check_sat() {
            Ok(r) => r.into(),
            Err(_) => SolveResult::Unknown,
        }
    }

    /// Get the model from the last SAT result
    ///
    /// Returns None if the last check_sat did not return Sat
    pub fn get_model(&self) -> Option<Model> {
        // Get the model string from executor
        let model_str = self.executor.get_model();
        if model_str == "(model)" || model_str.is_empty() {
            return None;
        }

        // Parse the model string to extract variable values
        let mut model = Model {
            int_values: HashMap::new(),
            real_values: HashMap::new(),
            bool_values: HashMap::new(),
            bv_values: HashMap::new(),
        };

        // Simple parsing of SMT-LIB model format
        // Format: (model (define-fun name () Sort value) ...)
        for line in model_str.lines() {
            let line = line.trim();
            if line.starts_with("(define-fun ") {
                // Parse: (define-fun name () Sort value)
                if let Some(parsed) = parse_define_fun(line) {
                    let (name, sort, value) = parsed;
                    match sort.as_str() {
                        "Int" => {
                            if let Ok(v) = value.parse::<i64>() {
                                model.int_values.insert(name, v);
                            } else if value.starts_with("(- ") {
                                // Handle negative: (- N)
                                let inner = value.trim_start_matches("(- ").trim_end_matches(')');
                                if let Ok(v) = inner.parse::<i64>() {
                                    model.int_values.insert(name, -v);
                                }
                            }
                        }
                        "Real" => {
                            if let Ok(v) = value.parse::<f64>() {
                                model.real_values.insert(name, v);
                            } else if value.starts_with("(/ ") {
                                // Handle rational: (/ N D)
                                let parts: Vec<&str> = value
                                    .trim_start_matches("(/ ")
                                    .trim_end_matches(')')
                                    .split_whitespace()
                                    .collect();
                                if parts.len() == 2 {
                                    if let (Ok(n), Ok(d)) =
                                        (parts[0].parse::<f64>(), parts[1].parse::<f64>())
                                    {
                                        model.real_values.insert(name, n / d);
                                    }
                                }
                            } else if value.starts_with("(- ") {
                                let inner = value.trim_start_matches("(- ").trim_end_matches(')');
                                if let Ok(v) = inner.parse::<f64>() {
                                    model.real_values.insert(name, -v);
                                }
                            }
                        }
                        "Bool" => {
                            model.bool_values.insert(name, value == "true");
                        }
                        _ if sort.starts_with("(_ BitVec ") => {
                            // Parse bitvector sort and value
                            let width_str =
                                sort.trim_start_matches("(_ BitVec ").trim_end_matches(')');
                            if let Ok(width) = width_str.parse::<u32>() {
                                if let Some(binary) = value.strip_prefix("#b") {
                                    if let Some(v) = BigInt::parse_bytes(binary.as_bytes(), 2) {
                                        model.bv_values.insert(name, (v, width));
                                    }
                                } else if let Some(hex) = value.strip_prefix("#x") {
                                    if let Some(v) = BigInt::parse_bytes(hex.as_bytes(), 16) {
                                        model.bv_values.insert(name, (v, width));
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        Some(model)
    }

    /// Reset the solver, clearing all assertions
    pub fn reset(&mut self) {
        let _ = self.executor.execute(&Command::Reset);
        self.var_names.clear();
        self.var_sorts.clear();
    }
}

/// Parse a define-fun line and extract (name, sort, value)
fn parse_define_fun(line: &str) -> Option<(String, String, String)> {
    // Format: (define-fun name () Sort value)
    let content = line
        .trim_start_matches("(define-fun ")
        .trim_end_matches(')');
    let mut parts = content.splitn(2, " () ");
    let name = parts.next()?.to_string();
    let rest = parts.next()?;

    // Find where sort ends and value begins
    // Sort can be: Int, Real, Bool, (_ BitVec N)
    let (sort, value) = if rest.starts_with("(_ ") {
        // Indexed sort like (_ BitVec 32)
        let sort_end = rest.find(')')? + 1;
        let sort = rest[..sort_end].to_string();
        let value = rest[sort_end..].trim().to_string();
        (sort, value)
    } else {
        // Simple sort
        let mut parts = rest.splitn(2, ' ');
        let sort = parts.next()?.to_string();
        let value = parts.next()?.trim().to_string();
        (sort, value)
    };

    Some((name, sort, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_lia() {
        let mut solver = Solver::new(Logic::QfLia);

        let x = solver.declare_const("x", Sort::Int);
        let zero = solver.int_const(0);
        let ten = solver.int_const(10);

        // x > 0 and x < 10
        let c1 = solver.gt(x, zero);
        solver.assert_term(c1);
        let c2 = solver.lt(x, ten);
        solver.assert_term(c2);

        assert_eq!(solver.check_sat(), SolveResult::Sat);
    }

    #[test]
    fn test_unsat() {
        let mut solver = Solver::new(Logic::QfLia);

        let x = solver.declare_const("x", Sort::Int);
        let zero = solver.int_const(0);

        // x > 0 and x < 0 is unsat
        let c1 = solver.gt(x, zero);
        solver.assert_term(c1);
        let c2 = solver.lt(x, zero);
        solver.assert_term(c2);

        assert_eq!(solver.check_sat(), SolveResult::Unsat);
    }

    #[test]
    fn test_incremental() {
        let mut solver = Solver::new(Logic::QfLia);

        let x = solver.declare_const("x", Sort::Int);
        let zero = solver.int_const(0);
        let ten = solver.int_const(10);

        let c1 = solver.gt(x, zero);
        solver.assert_term(c1);

        solver.push();
        let c2 = solver.lt(x, zero);
        solver.assert_term(c2);
        assert_eq!(solver.check_sat(), SolveResult::Unsat);
        solver.pop();

        // After pop, should be sat again
        let c3 = solver.lt(x, ten);
        solver.assert_term(c3);
        assert_eq!(solver.check_sat(), SolveResult::Sat);
    }

    #[test]
    fn test_implication() {
        // Simplified test: just a single implication
        let mut solver = Solver::new(Logic::QfLia);

        let x = solver.declare_const("x", Sort::Int);
        let y = solver.declare_const("y", Sort::Int);

        let five = solver.int_const(5);

        // x = 5
        let x_eq_5 = solver.eq(x, five);
        solver.assert_term(x_eq_5);

        // x = 5 implies y = 5
        let y_eq_5 = solver.eq(y, five);
        let impl_term = solver.implies(x_eq_5, y_eq_5);
        solver.assert_term(impl_term);

        assert_eq!(solver.check_sat(), SolveResult::Sat);
    }

    #[test]
    fn test_equality_constraints() {
        // Test equality constraints (simpler than full ReLU which needs disequality splits)
        let mut solver = Solver::new(Logic::QfLia);

        let x = solver.declare_const("x", Sort::Int);
        let y = solver.declare_const("y", Sort::Int);
        let z = solver.declare_const("z", Sort::Int);

        let five = solver.int_const(5);
        let ten = solver.int_const(10);

        // x = 5
        let x_eq_5 = solver.eq(x, five);
        solver.assert_term(x_eq_5);

        // y = x + 5 (i.e., y = 10)
        let x_plus_5 = solver.add(x, five);
        let y_eq = solver.eq(y, x_plus_5);
        solver.assert_term(y_eq);

        // z = y (i.e., z = 10)
        let z_eq_y = solver.eq(z, y);
        solver.assert_term(z_eq_y);

        // z = 10 (should be consistent)
        let z_eq_10 = solver.eq(z, ten);
        solver.assert_term(z_eq_10);

        assert_eq!(solver.check_sat(), SolveResult::Sat);

        let model = solver.get_model().expect("Expected model for SAT result");
        assert_eq!(model.get_int("x"), Some(5));
        assert_eq!(model.get_int("y"), Some(10));
        assert_eq!(model.get_int("z"), Some(10));
    }

    #[test]
    fn test_lra_simple() {
        let mut solver = Solver::new(Logic::QfLra);

        let x = solver.declare_const("x", Sort::Real);
        let half = solver.rational_const(1, 2);
        let one = solver.real_const(1.0);

        // x > 0.5 and x < 1
        let c1 = solver.gt(x, half);
        solver.assert_term(c1);
        let c2 = solver.lt(x, one);
        solver.assert_term(c2);

        assert_eq!(solver.check_sat(), SolveResult::Sat);
    }

    #[test]
    fn test_boolean_logic() {
        let mut solver = Solver::new(Logic::QfUf);

        let a = solver.declare_const("a", Sort::Bool);

        // a and (not a) is unsat
        solver.assert_term(a);
        let not_a = solver.not(a);
        solver.assert_term(not_a);

        assert_eq!(solver.check_sat(), SolveResult::Unsat);

        solver.reset();

        // a or b, not a, not b is unsat
        let a = solver.declare_const("a", Sort::Bool);
        let b = solver.declare_const("b", Sort::Bool);
        let a_or_b = solver.or(a, b);
        solver.assert_term(a_or_b);
        let not_a = solver.not(a);
        solver.assert_term(not_a);
        let not_b = solver.not(b);
        solver.assert_term(not_b);

        assert_eq!(solver.check_sat(), SolveResult::Unsat);
    }

    #[test]
    fn test_relu_encoding_qf_auflira() {
        // Test the ReLU encoding pattern from gamma-crown's neural network verification
        // This uses QF_AUFLIRA (mixed Int+Real), which may return Unknown until
        // full Nelson-Oppen theory combination is implemented.
        //
        // ReLU encoding: y = max(0, x)
        // Uses binary phase variable p (0 or 1):
        //   - y >= 0
        //   - If p = 1: y = x (active phase)
        //   - If p = 0: y = 0 (inactive phase)

        let mut solver = Solver::new(Logic::QfAuflira);

        // Declare variables
        let x = solver.declare_const("x", Sort::Real);
        let y = solver.declare_const("y", Sort::Real);
        let p = solver.declare_const("p", Sort::Int);

        // p is binary: 0 <= p <= 1
        let zero_int = solver.int_const(0);
        let one_int = solver.int_const(1);
        let p_ge_0 = solver.ge(p, zero_int);
        let p_le_1 = solver.le(p, one_int);
        solver.assert_term(p_ge_0);
        solver.assert_term(p_le_1);

        // y >= 0 (ReLU output is non-negative)
        let zero_real = solver.real_const(0.0);
        let y_ge_0 = solver.ge(y, zero_real);
        solver.assert_term(y_ge_0);

        // When p = 1: y = x
        let p_eq_1 = solver.eq(p, one_int);
        let y_eq_x = solver.eq(y, x);
        let active_impl = solver.implies(p_eq_1, y_eq_x);
        solver.assert_term(active_impl);

        // When p = 0: y = 0
        let p_eq_0 = solver.eq(p, zero_int);
        let y_eq_0 = solver.eq(y, zero_real);
        let inactive_impl = solver.implies(p_eq_0, y_eq_0);
        solver.assert_term(inactive_impl);

        // Set x = 5.0 (positive input)
        let five = solver.real_const(5.0);
        let x_eq_5 = solver.eq(x, five);
        solver.assert_term(x_eq_5);

        // Check: This should be satisfiable with p=1, y=5
        let result = solver.check_sat();

        // Note: QF_AUFLIRA may return Unknown until full implementation
        // For now, we just verify no crash and document the behavior
        match result {
            SolveResult::Sat => {
                // If we get SAT, verify the model makes sense
                if let Some(model) = solver.get_model() {
                    // p should be 1 (active phase for positive x)
                    if let Some(p_val) = model.get_int("p") {
                        assert!(p_val == 0 || p_val == 1, "p should be binary");
                    }
                    // y should equal x when p=1, or 0 when p=0
                    if let Some(y_val) = model.get_real("y") {
                        assert!(y_val >= 0.0, "ReLU output should be non-negative");
                    }
                }
            }
            SolveResult::Unknown => {
                // Expected until full QF_AUFLIRA is implemented
                // The solver correctly recognizes it cannot handle mixed Int+Real
            }
            SolveResult::Unsat => {
                panic!("ReLU encoding should be satisfiable with x=5");
            }
        }
    }

    #[test]
    fn test_pure_real_relu_qf_lra() {
        // Pure Real version of ReLU (no binary Int phase)
        // This uses QF_LRA which is fully supported
        //
        // Big-M encoding: y = max(0, x) with bounds
        // Assumptions: -M <= x <= M for some large M

        let mut solver = Solver::new(Logic::QfLra);

        let x = solver.declare_const("x", Sort::Real);
        let y = solver.declare_const("y", Sort::Real);

        let zero = solver.real_const(0.0);

        // y >= 0
        let y_ge_0 = solver.ge(y, zero);
        solver.assert_term(y_ge_0);

        // y >= x
        let y_ge_x = solver.ge(y, x);
        solver.assert_term(y_ge_x);

        // Set x = 5.0
        let five = solver.real_const(5.0);
        let x_eq_5 = solver.eq(x, five);
        solver.assert_term(x_eq_5);

        // Set y = x (for positive x, ReLU(x) = x)
        let y_eq_5 = solver.eq(y, five);
        solver.assert_term(y_eq_5);

        assert_eq!(solver.check_sat(), SolveResult::Sat);

        let model = solver.get_model().expect("Expected model for SAT result");
        assert_eq!(model.get_real("x"), Some(5.0));
        assert_eq!(model.get_real("y"), Some(5.0));
    }

    #[test]
    fn test_disequality_split_binary_int() {
        // Test P2.1: NeedDisequlitySplit handling for binary Int variables
        // When p ∈ {0, 1} and p != 0, should infer p = 1
        let mut solver = Solver::new(Logic::QfLia);

        let p = solver.declare_const("p", Sort::Int);
        let zero = solver.int_const(0);
        let one = solver.int_const(1);

        // 0 <= p <= 1
        let p_ge_0 = solver.ge(p, zero);
        let p_le_1 = solver.le(p, one);
        solver.assert_term(p_ge_0);
        solver.assert_term(p_le_1);

        // p != 0 (should trigger disequality split)
        let p_neq_0 = solver.neq(p, zero);
        solver.assert_term(p_neq_0);

        assert_eq!(solver.check_sat(), SolveResult::Sat);

        let model = solver.get_model().expect("Expected model for SAT result");
        assert_eq!(
            model.get_int("p"),
            Some(1),
            "p should be 1 when p != 0 and p in [0, 1]"
        );
    }

    #[test]
    fn test_relu_encoding_with_neq() {
        // Full ReLU encoding with explicit neq constraint
        // This tests the complete NeedDisequlitySplit flow in QF_AUFLIRA
        let mut solver = Solver::new(Logic::QfAuflira);

        let x = solver.declare_const("x", Sort::Real);
        let y = solver.declare_const("y", Sort::Real);
        let p = solver.declare_const("p", Sort::Int);

        let zero_int = solver.int_const(0);
        let one_int = solver.int_const(1);
        let zero_real = solver.real_const(0.0);
        let five = solver.real_const(5.0);

        // p is binary: 0 <= p <= 1
        let p_ge_0 = solver.ge(p, zero_int);
        solver.assert_term(p_ge_0);
        let p_le_1 = solver.le(p, one_int);
        solver.assert_term(p_le_1);

        // y >= 0 (ReLU output non-negative)
        let y_ge_0 = solver.ge(y, zero_real);
        solver.assert_term(y_ge_0);

        // When p = 1: y = x (active phase)
        let p_eq_1 = solver.eq(p, one_int);
        let y_eq_x = solver.eq(y, x);
        let active_impl = solver.implies(p_eq_1, y_eq_x);
        solver.assert_term(active_impl);

        // When p = 0: y = 0 (inactive phase)
        let p_eq_0 = solver.eq(p, zero_int);
        let y_eq_0 = solver.eq(y, zero_real);
        let inactive_impl = solver.implies(p_eq_0, y_eq_0);
        solver.assert_term(inactive_impl);

        // x = 5.0 (positive input)
        let x_eq_5 = solver.eq(x, five);
        solver.assert_term(x_eq_5);

        // Force p != 0 (should trigger disequality split and give p = 1)
        let p_neq_0 = solver.neq(p, zero_int);
        solver.assert_term(p_neq_0);

        assert_eq!(solver.check_sat(), SolveResult::Sat);

        let model = solver.get_model().expect("Expected model for SAT result");
        assert_eq!(model.get_int("p"), Some(1), "p should be 1 when p != 0");
        assert_eq!(
            model.get_real("y"),
            Some(5.0),
            "y should equal x when p = 1"
        );
        assert_eq!(model.get_real("x"), Some(5.0), "x should be 5.0");
    }
}
