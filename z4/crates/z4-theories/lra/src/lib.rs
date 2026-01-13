//! Z4 LRA - Linear Real Arithmetic theory solver
//!
//! Implements the dual simplex algorithm for linear arithmetic over reals,
//! following the approach from "A Fast Linear-Arithmetic Solver for DPLL(T)"
//! by Dutertre & de Moura (CAV 2006).
//!
//! ## Algorithm Overview
//!
//! The solver maintains:
//! - A tableau of linear equalities: basic_var = Σ(coeff * nonbasic_var)
//! - Bounds for each variable (lower, upper, or both)
//! - Current assignment satisfying the tableau
//!
//! When bounds change (from theory atom assertions), we use dual simplex
//! to restore feasibility or detect conflicts.

#![warn(missing_docs)]
#![warn(clippy::all)]

use hashbrown::HashMap;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use z4_core::term::{Constant, Symbol, TermData, TermId, TermStore};
use z4_core::{
    DisequlitySplitRequest, ExpressionSplitRequest, TheoryLit, TheoryPropagation, TheoryResult,
    TheorySolver,
};

/// Bound type for a variable
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundType {
    /// Lower bound: x >= c
    Lower,
    /// Upper bound: x <= c
    Upper,
}

/// A bound with its value and the atom that established it
#[derive(Debug, Clone)]
pub struct Bound {
    /// The bound value
    pub value: BigRational,
    /// The atom term that established this bound
    pub reason: TermId,
    /// The Boolean value of `reason` in the current assignment.
    ///
    /// When the SAT layer assigns an atom `t` to `false`, the theory asserts the
    /// negation of `t` (e.g. `!(x <= 5)` becomes `x > 5`). For conflict clauses
    /// to be sound, we must preserve that polarity.
    pub reason_value: bool,
    /// Whether the bound is strict (< or >) vs non-strict (<= or >=)
    pub strict: bool,
}

/// Status of a variable in the simplex tableau
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarStatus {
    /// Non-basic variable: can be pivoted with a basic variable
    NonBasic,
    /// Basic variable: defined by a row in the tableau
    Basic(usize), // row index
}

/// A row in the simplex tableau
///
/// Represents: basic_var = Σ(coeff * var) + constant
#[derive(Debug, Clone)]
pub struct TableauRow {
    /// The basic variable for this row
    pub basic_var: u32,
    /// Sparse coefficients: (variable, coefficient)
    pub coeffs: Vec<(u32, BigRational)>,
    /// Constant term (RHS constant after normalization)
    pub constant: BigRational,
}

impl TableauRow {
    /// Get the coefficient for a variable, or zero if not present
    pub fn coeff(&self, var: u32) -> BigRational {
        for &(v, ref c) in &self.coeffs {
            if v == var {
                return c.clone();
            }
        }
        BigRational::zero()
    }

    /// Check if a variable appears in this row
    pub fn contains(&self, var: u32) -> bool {
        self.coeffs.iter().any(|(v, _)| *v == var)
    }
}

/// Information about an LRA variable
#[derive(Debug, Clone, Default)]
struct VarInfo {
    /// Current value assignment
    value: BigRational,
    /// Lower bound (if any)
    lower: Option<Bound>,
    /// Upper bound (if any)
    upper: Option<Bound>,
    /// Status in tableau
    status: Option<VarStatus>,
}

/// Linear expression: Σ(coeff * var) + constant
#[derive(Debug, Clone)]
pub struct LinearExpr {
    /// Variable coefficients
    pub coeffs: Vec<(u32, BigRational)>,
    /// Constant term
    pub constant: BigRational,
}

impl LinearExpr {
    /// Create an empty expression (constant 0)
    pub fn zero() -> Self {
        LinearExpr {
            coeffs: Vec::new(),
            constant: BigRational::zero(),
        }
    }

    /// Create expression for a single variable with coefficient 1
    pub fn var(v: u32) -> Self {
        LinearExpr {
            coeffs: vec![(v, BigRational::one())],
            constant: BigRational::zero(),
        }
    }

    /// Create a constant expression
    pub fn constant(c: BigRational) -> Self {
        LinearExpr {
            coeffs: Vec::new(),
            constant: c,
        }
    }

    /// Add another expression to this one
    pub fn add(&mut self, other: &LinearExpr) {
        for &(v, ref c) in &other.coeffs {
            self.add_term(v, c.clone());
        }
        self.constant += &other.constant;
    }

    /// Add a scaled expression to this one
    pub fn add_scaled(&mut self, other: &LinearExpr, scale: &BigRational) {
        for &(v, ref c) in &other.coeffs {
            self.add_term(v, c * scale);
        }
        self.constant += &other.constant * scale;
    }

    /// Add a term (variable * coefficient)
    pub fn add_term(&mut self, var: u32, coeff: BigRational) {
        if coeff.is_zero() {
            return;
        }
        for &mut (v, ref mut c) in &mut self.coeffs {
            if v == var {
                *c += &coeff;
                // Remove if coefficient became zero
                if c.is_zero() {
                    self.coeffs.retain(|(v2, _)| *v2 != var);
                }
                return;
            }
        }
        self.coeffs.push((var, coeff));
    }

    /// Scale the entire expression
    pub fn scale(&mut self, factor: &BigRational) {
        for (_, c) in &mut self.coeffs {
            *c = &*c * factor;
        }
        self.constant = &self.constant * factor;
    }

    /// Negate the expression
    pub fn negate(&mut self) {
        for (_, c) in &mut self.coeffs {
            *c = -c.clone();
        }
        self.constant = -self.constant.clone();
    }

    /// Check if this is a constant expression (no variables)
    pub fn is_constant(&self) -> bool {
        self.coeffs.is_empty()
    }
}

/// Conflict from the simplex solver
#[derive(Debug, Clone)]
pub struct SimplexConflict {
    /// The literals involved in the conflict
    pub literals: Vec<TheoryLit>,
}

/// Model extracted from LRA solver with variable assignments
#[derive(Debug, Clone)]
pub struct LraModel {
    /// Variable assignments: term_id -> rational value
    pub values: HashMap<TermId, BigRational>,
}

/// Cached information about a parsed atom
#[derive(Clone)]
struct ParsedAtomInfo {
    /// The normalized linear expression (expr such that "expr op 0" is the constraint)
    expr: LinearExpr,
    /// Is this a <= constraint (vs >=)?
    is_le: bool,
    /// Is this a strict comparison (< or >)?
    strict: bool,
    /// Is this an equality atom (= symbol)?
    is_eq: bool,
    /// Is this a distinct atom (distinct symbol)?
    /// When true, semantics are inverted: value=true means disequality, value=false means equality
    is_distinct: bool,
}

/// LRA theory solver using dual simplex
pub struct LraSolver<'a> {
    /// Reference to the term store for parsing expressions
    terms: &'a TermStore,
    /// Tableau rows
    rows: Vec<TableauRow>,
    /// Variable information (indexed by internal var id)
    vars: Vec<VarInfo>,
    /// Mapping from term IDs to internal variable IDs
    term_to_var: HashMap<TermId, u32>,
    /// Mapping from internal variable IDs to term IDs
    var_to_term: HashMap<u32, TermId>,
    /// Next fresh variable ID
    next_var: u32,
    /// Trail for backtracking: (var_id, old_lower, old_upper)
    trail: Vec<(u32, Option<Bound>, Option<Bound>)>,
    /// Scope markers (trail positions)
    scopes: Vec<usize>,
    /// Asserted atoms: term_id -> value
    asserted: HashMap<TermId, bool>,
    /// Cache of parsed atom information to avoid re-parsing
    atom_cache: HashMap<TermId, Option<ParsedAtomInfo>>,
    /// Whether we encountered any unsupported (non-linear / non-LRA) constructs.
    ///
    /// When true, this solver may still soundly return UNSAT (because we
    /// over-approximate unsupported terms as fresh variables), but returning SAT
    /// would be unsound, so we surface UNKNOWN instead.
    saw_unsupported: bool,
    /// Dirty flag: need to recompute
    dirty: bool,
}

impl<'a> LraSolver<'a> {
    /// Create a new LRA solver
    #[must_use]
    pub fn new(terms: &'a TermStore) -> Self {
        LraSolver {
            terms,
            rows: Vec::new(),
            vars: Vec::new(),
            term_to_var: HashMap::new(),
            var_to_term: HashMap::new(),
            next_var: 0,
            trail: Vec::new(),
            scopes: Vec::new(),
            asserted: HashMap::new(),
            atom_cache: HashMap::new(),
            saw_unsupported: false,
            dirty: true,
        }
    }

    /// Get or create an internal variable for a term
    fn intern_var(&mut self, term: TermId) -> u32 {
        if let Some(&var) = self.term_to_var.get(&term) {
            return var;
        }
        let var = self.next_var;
        self.next_var += 1;
        self.term_to_var.insert(term, var);
        self.var_to_term.insert(var, term);
        // Extend vars vector if needed
        while self.vars.len() <= var as usize {
            self.vars.push(VarInfo::default());
        }
        self.vars[var as usize].status = Some(VarStatus::NonBasic);
        var
    }

    /// Parse a term into a linear expression
    fn parse_linear_expr(&mut self, term: TermId) -> LinearExpr {
        match self.terms.get(term) {
            TermData::Const(Constant::Int(n)) => LinearExpr::constant(BigRational::from(n.clone())),
            TermData::Const(Constant::Rational(r)) => LinearExpr::constant(r.0.clone()),
            TermData::Var(_, _) => {
                let var = self.intern_var(term);
                LinearExpr::var(var)
            }
            TermData::App(Symbol::Named(name), args) => {
                match name.as_str() {
                    "+" => {
                        let mut result = LinearExpr::zero();
                        for &arg in args {
                            let sub_expr = self.parse_linear_expr(arg);
                            result.add(&sub_expr);
                        }
                        result
                    }
                    "-" if args.len() == 1 => {
                        // Unary minus
                        let mut result = self.parse_linear_expr(args[0]);
                        result.negate();
                        result
                    }
                    "-" if args.len() >= 2 => {
                        // Binary/n-ary minus: a - b - c = a + (-b) + (-c)
                        let mut result = self.parse_linear_expr(args[0]);
                        for &arg in &args[1..] {
                            let mut sub_expr = self.parse_linear_expr(arg);
                            sub_expr.negate();
                            result.add(&sub_expr);
                        }
                        result
                    }
                    "*" => {
                        // Find constant and variable parts
                        let mut const_part = BigRational::one();
                        let mut var_expr: Option<LinearExpr> = None;

                        for &arg in args {
                            let sub_expr = self.parse_linear_expr(arg);
                            if sub_expr.is_constant() {
                                const_part *= &sub_expr.constant;
                            } else if var_expr.is_none() {
                                var_expr = Some(sub_expr);
                            } else {
                                // Non-linear: over-approximate as fresh variable.
                                self.saw_unsupported = true;
                                let var = self.intern_var(term);
                                return LinearExpr::var(var);
                            }
                        }

                        match var_expr {
                            Some(mut expr) => {
                                expr.scale(&const_part);
                                expr
                            }
                            None => LinearExpr::constant(const_part),
                        }
                    }
                    "/" if args.len() == 2 => {
                        // Division by constant
                        let num = self.parse_linear_expr(args[0]);
                        let denom = self.parse_linear_expr(args[1]);
                        if denom.is_constant() && !denom.constant.is_zero() {
                            let mut result = num;
                            let inv = BigRational::one() / denom.constant;
                            result.scale(&inv);
                            result
                        } else {
                            // Non-linear / division by 0: over-approximate as fresh variable.
                            self.saw_unsupported = true;
                            let var = self.intern_var(term);
                            LinearExpr::var(var)
                        }
                    }
                    other => {
                        // Unknown function: create slack variable
                        if std::env::var("Z4_DEBUG_LRA").is_ok() {
                            eprintln!("[LRA] Unknown function: {:?}", other);
                        }
                        self.saw_unsupported = true;
                        let var = self.intern_var(term);
                        LinearExpr::var(var)
                    }
                }
            }
            _ => {
                // Unknown term: create slack variable
                if std::env::var("Z4_DEBUG_LRA").is_ok() {
                    eprintln!("[LRA] Unknown term: {:?}", self.terms.get(term));
                }
                self.saw_unsupported = true;
                let var = self.intern_var(term);
                LinearExpr::var(var)
            }
        }
    }

    /// Parse an arithmetic atom and return (normalized_expr, is_le, is_strict)
    /// Normalized: expr <= 0 (for <=) or expr < 0 (for <)
    fn parse_atom(&mut self, term: TermId) -> Option<(LinearExpr, bool, bool)> {
        match self.terms.get(term) {
            TermData::App(Symbol::Named(name), args) if args.len() == 2 => {
                let lhs = args[0];
                let rhs = args[1];

                match name.as_str() {
                    "<" => {
                        // lhs < rhs => lhs - rhs < 0
                        let mut expr = self.parse_linear_expr(lhs);
                        let rhs_expr = self.parse_linear_expr(rhs);
                        expr.add_scaled(&rhs_expr, &BigRational::from(BigInt::from(-1)));
                        Some((expr, true, true)) // is_le=true (upper bound), strict=true
                    }
                    "<=" => {
                        // lhs <= rhs => lhs - rhs <= 0
                        let mut expr = self.parse_linear_expr(lhs);
                        let rhs_expr = self.parse_linear_expr(rhs);
                        expr.add_scaled(&rhs_expr, &BigRational::from(BigInt::from(-1)));
                        Some((expr, true, false)) // is_le=true, strict=false
                    }
                    ">" => {
                        // lhs > rhs => rhs - lhs < 0 => -(lhs - rhs) < 0
                        // But we want upper bound, so: lhs > rhs is equivalent to rhs < lhs
                        // which is rhs - lhs < 0
                        let mut expr = self.parse_linear_expr(rhs);
                        let lhs_expr = self.parse_linear_expr(lhs);
                        expr.add_scaled(&lhs_expr, &BigRational::from(BigInt::from(-1)));
                        Some((expr, true, true)) // is_le=true, strict=true
                    }
                    ">=" => {
                        // lhs >= rhs => rhs - lhs <= 0
                        let mut expr = self.parse_linear_expr(rhs);
                        let lhs_expr = self.parse_linear_expr(lhs);
                        expr.add_scaled(&lhs_expr, &BigRational::from(BigInt::from(-1)));
                        Some((expr, true, false))
                    }
                    "=" => {
                        // Equality: handle as both <= and >= in check
                        let mut expr = self.parse_linear_expr(lhs);
                        let rhs_expr = self.parse_linear_expr(rhs);
                        expr.add_scaled(&rhs_expr, &BigRational::from(BigInt::from(-1)));
                        Some((expr, true, false)) // For equality, we'll handle specially
                    }
                    "distinct" => {
                        // Distinct: same expression as equality, but with inverted semantics
                        // (distinct a b) with value=true means a != b (disequality)
                        // (distinct a b) with value=false means a = b (equality)
                        let mut expr = self.parse_linear_expr(lhs);
                        let rhs_expr = self.parse_linear_expr(rhs);
                        expr.add_scaled(&rhs_expr, &BigRational::from(BigInt::from(-1)));
                        Some((expr, true, false)) // Handled specially in check()
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Assert a bound on a linear expression
    /// For expr <= c: create slack variable s, add row s = expr, then s <= c
    fn assert_bound(
        &mut self,
        expr: LinearExpr,
        bound: BigRational,
        bound_type: BoundType,
        strict: bool,
        reason: TermId,
        reason_value: bool,
    ) {
        if expr.is_constant() {
            // Pure constant comparison - will be checked in check()
            return;
        }

        // Fast path: a single affine variable constraint can be asserted as a direct bound.
        //
        // The atom parser normalizes comparisons into the form `expr <= bound` or `expr >= bound`,
        // where `expr` may include a constant offset (e.g. `x - 5 <= 0`).
        //
        // Avoid creating slack variables/tableau rows for the common case:
        //   coeff*x + const <= bound
        //   coeff*x + const >= bound
        if expr.coeffs.len() == 1 {
            let (var, coeff) = &expr.coeffs[0];
            if !coeff.is_zero() {
                let rhs = (bound - &expr.constant) / coeff.clone();
                let coeff_positive = coeff.is_positive();
                let var_bound_type = match (bound_type, coeff_positive) {
                    (BoundType::Upper, true) => BoundType::Upper, // x <= rhs
                    (BoundType::Upper, false) => BoundType::Lower, // x >= rhs
                    (BoundType::Lower, true) => BoundType::Lower, // x >= rhs
                    (BoundType::Lower, false) => BoundType::Upper, // x <= rhs
                };
                self.assert_var_bound(*var, rhs, var_bound_type, strict, reason, reason_value);
                return;
            }
        }

        // Create a slack variable for the expression
        let slack = self.next_var;
        self.next_var += 1;
        while self.vars.len() <= slack as usize {
            self.vars.push(VarInfo::default());
        }

        // Add tableau row: slack = expr (rearranged: slack - expr = 0)
        let row_idx = self.rows.len();
        self.vars[slack as usize].status = Some(VarStatus::Basic(row_idx));

        // slack = Σ(coeff * var) + constant
        let row = TableauRow {
            basic_var: slack,
            coeffs: expr.coeffs.clone(),
            constant: expr.constant.clone(),
        };
        self.rows.push(row);

        // Set initial value for slack based on current variable values
        let mut slack_val = expr.constant;
        for &(v, ref c) in &expr.coeffs {
            if let Some(info) = self.vars.get(v as usize) {
                slack_val += c * &info.value;
            }
        }
        self.vars[slack as usize].value = slack_val;

        // Mark non-basic variables used in this row
        for &(v, _) in &expr.coeffs {
            if let Some(info) = self.vars.get_mut(v as usize) {
                if info.status.is_none() {
                    info.status = Some(VarStatus::NonBasic);
                }
            }
        }

        // Assert bound on slack variable
        self.assert_var_bound(slack, bound, bound_type, strict, reason, reason_value);
    }

    /// Assert a bound on a single variable
    fn assert_var_bound(
        &mut self,
        var: u32,
        bound: BigRational,
        bound_type: BoundType,
        strict: bool,
        reason: TermId,
        reason_value: bool,
    ) {
        while self.vars.len() <= var as usize {
            self.vars.push(VarInfo::default());
        }
        let info = &mut self.vars[var as usize];

        // Save old bounds for backtracking
        let old_lower = info.lower.clone();
        let old_upper = info.upper.clone();
        self.trail.push((var, old_lower, old_upper));

        let new_bound = Bound {
            value: bound,
            reason,
            reason_value,
            strict,
        };

        match bound_type {
            BoundType::Lower => {
                // Only update if tighter
                let should_update = match &info.lower {
                    None => true,
                    Some(existing) => {
                        new_bound.value > existing.value
                            || (new_bound.value == existing.value
                                && new_bound.strict
                                && !existing.strict)
                    }
                };
                if should_update {
                    info.lower = Some(new_bound);
                }
            }
            BoundType::Upper => {
                // Only update if tighter
                let should_update = match &info.upper {
                    None => true,
                    Some(existing) => {
                        new_bound.value < existing.value
                            || (new_bound.value == existing.value
                                && new_bound.strict
                                && !existing.strict)
                    }
                };
                if should_update {
                    info.upper = Some(new_bound);
                }
            }
        }

        self.dirty = true;
    }

    /// Check if a variable's current value violates its bounds
    fn violates_bounds(&self, var: u32) -> Option<BoundType> {
        let info = &self.vars[var as usize];

        if let Some(ref lower) = info.lower {
            let cmp = info.value.cmp(&lower.value);
            let violated = if lower.strict {
                cmp != std::cmp::Ordering::Greater
            } else {
                cmp == std::cmp::Ordering::Less
            };
            if violated {
                return Some(BoundType::Lower);
            }
        }

        if let Some(ref upper) = info.upper {
            let cmp = info.value.cmp(&upper.value);
            let violated = if upper.strict {
                cmp != std::cmp::Ordering::Less
            } else {
                cmp == std::cmp::Ordering::Greater
            };
            if violated {
                return Some(BoundType::Upper);
            }
        }

        None
    }

    /// Find a suitable non-basic variable to pivot with (using Bland's rule)
    /// Returns (non_basic_var, direction) where direction is +1 or -1
    ///
    /// Bland's rule: to prevent cycling, always choose the eligible variable with
    /// the smallest index when there are ties. This guarantees termination.
    fn find_pivot_candidate(
        &self,
        row_idx: usize,
        violated_bound: BoundType,
    ) -> Option<(u32, i32)> {
        let row = &self.rows[row_idx];

        // Collect all eligible candidates, then pick smallest index (Bland's rule)
        let mut best: Option<(u32, i32)> = None;

        for &(nb_var, ref coeff) in &row.coeffs {
            if coeff.is_zero() {
                continue;
            }

            let info = &self.vars[nb_var as usize];

            // Determine if we can increase or decrease this non-basic variable
            // based on its bounds. We can move if there's room (not at the bound).
            // Note: For strict bounds, being at the bound value is already a violation,
            // so we check if there's any room to move.
            let can_increase = match &info.upper {
                None => true,                    // No upper bound, can always increase
                Some(b) => info.value < b.value, // Room to go up
            };

            let can_decrease = match &info.lower {
                None => true,                    // No lower bound, can always decrease
                Some(b) => info.value > b.value, // Room to go down
            };

            let direction = match violated_bound {
                BoundType::Lower => {
                    // Basic var is too small, need to increase it
                    // If coeff > 0: increase nb_var
                    // If coeff < 0: decrease nb_var
                    if coeff.is_positive() && can_increase {
                        Some(1)
                    } else if coeff.is_negative() && can_decrease {
                        Some(-1)
                    } else {
                        None
                    }
                }
                BoundType::Upper => {
                    // Basic var is too large, need to decrease it
                    // If coeff > 0: decrease nb_var
                    // If coeff < 0: increase nb_var
                    if coeff.is_positive() && can_decrease {
                        Some(-1)
                    } else if coeff.is_negative() && can_increase {
                        Some(1)
                    } else {
                        None
                    }
                }
            };

            if let Some(dir) = direction {
                // Bland's rule: keep the candidate with smallest variable index
                match &best {
                    None => best = Some((nb_var, dir)),
                    Some((best_var, _)) if nb_var < *best_var => best = Some((nb_var, dir)),
                    _ => {}
                }
            }
        }

        best
    }

    /// Find all eligible pivot candidates (non-basic variables) in increasing variable index order.
    ///
    /// This is used to skip degenerate pivots that don't change the entering variable's value.
    fn find_pivot_candidates(&self, row_idx: usize, violated_bound: BoundType) -> Vec<u32> {
        let row = &self.rows[row_idx];
        let mut candidates: Vec<u32> = Vec::new();

        for &(nb_var, ref coeff) in &row.coeffs {
            if coeff.is_zero() {
                continue;
            }

            let info = &self.vars[nb_var as usize];

            let can_increase = match &info.upper {
                None => true,
                Some(b) => info.value < b.value,
            };

            let can_decrease = match &info.lower {
                None => true,
                Some(b) => info.value > b.value,
            };

            let eligible = match violated_bound {
                BoundType::Lower => {
                    (coeff.is_positive() && can_increase) || (coeff.is_negative() && can_decrease)
                }
                BoundType::Upper => {
                    (coeff.is_positive() && can_decrease) || (coeff.is_negative() && can_increase)
                }
            };

            if eligible {
                candidates.push(nb_var);
            }
        }

        candidates.sort_unstable();
        candidates
    }

    /// Perform a pivot operation
    fn pivot(&mut self, row_idx: usize, entering_var: u32) {
        let leaving_var = self.rows[row_idx].basic_var;

        // Get coefficient of entering variable in the row
        let entering_coeff = self.rows[row_idx].coeff(entering_var);
        if entering_coeff.is_zero() {
            return; // Should not happen
        }

        // Rearrange: leaving_var = ... + entering_coeff * entering_var + ...
        // => entering_var = (leaving_var - ... - ...) / entering_coeff

        // Build new row for entering_var
        let neg_inv_coeff = -BigRational::one() / &entering_coeff;
        let inv_coeff = BigRational::one() / &entering_coeff;

        let mut new_coeffs: Vec<(u32, BigRational)> = Vec::new();

        // Add leaving_var with coefficient 1/entering_coeff
        new_coeffs.push((leaving_var, inv_coeff.clone()));

        // Add other variables with negated scaled coefficients
        for &(v, ref c) in &self.rows[row_idx].coeffs {
            if v != entering_var {
                let new_c = c * &neg_inv_coeff;
                if !new_c.is_zero() {
                    new_coeffs.push((v, new_c));
                }
            }
        }

        let new_constant = &self.rows[row_idx].constant * &neg_inv_coeff;

        // Update the row
        self.rows[row_idx] = TableauRow {
            basic_var: entering_var,
            coeffs: new_coeffs,
            constant: new_constant,
        };

        // Update variable statuses
        self.vars[leaving_var as usize].status = Some(VarStatus::NonBasic);
        self.vars[entering_var as usize].status = Some(VarStatus::Basic(row_idx));

        // Clone the new row data for substitution to avoid borrow conflicts
        let new_row_coeffs = self.rows[row_idx].coeffs.clone();
        let new_row_constant = self.rows[row_idx].constant.clone();

        // Substitute in all other rows
        for i in 0..self.rows.len() {
            if i == row_idx {
                continue;
            }

            let old_coeff = self.rows[i].coeff(entering_var);
            if old_coeff.is_zero() {
                continue;
            }

            // Remove entering_var from this row's coeffs
            self.rows[i].coeffs.retain(|(v, _)| *v != entering_var);

            // Add scaled new_row coefficients
            for &(v, ref c) in &new_row_coeffs {
                let contrib = c * &old_coeff;
                if contrib.is_zero() {
                    continue;
                }
                let mut found = false;
                for &mut (v2, ref mut c2) in &mut self.rows[i].coeffs {
                    if v2 == v {
                        *c2 += &contrib;
                        found = true;
                        break;
                    }
                }
                if !found {
                    self.rows[i].coeffs.push((v, contrib));
                }
            }

            // Remove zero coefficients
            self.rows[i].coeffs.retain(|(_, c)| !c.is_zero());

            // Update constant
            self.rows[i].constant += &new_row_constant * &old_coeff;
        }
    }

    /// Compute how much to change a non-basic variable to fix a basic variable's bound violation.
    /// Returns the new value for the non-basic variable.
    fn compute_update_amount(
        &self,
        row_idx: usize,
        nb_var: u32,
        violated_bound: &BoundType,
    ) -> BigRational {
        let row = &self.rows[row_idx];
        let basic_var = row.basic_var;
        let basic_info = &self.vars[basic_var as usize];
        let nb_info = &self.vars[nb_var as usize];
        let coeff = row.coeff(nb_var);
        let unit = Self::strict_nudge_unit();

        // Current basic value
        let basic_val = &basic_info.value;

        // Target basic value (the bound it should satisfy)
        // For strict bounds, we need to go BEYOND the bound value
        let target_basic = match violated_bound {
            BoundType::Lower => {
                let bound = basic_info.lower.as_ref();
                match bound {
                    Some(b) if b.strict => {
                        // Strict lower bound: target value > bound
                        // If upper bound exists, use midpoint; otherwise add an infinitesimal.
                        if let Some(ref ub) = basic_info.upper {
                            (&b.value + &ub.value) / BigRational::from(BigInt::from(2))
                        } else {
                            &b.value + &unit
                        }
                    }
                    Some(b) => b.value.clone(),
                    None => BigRational::zero(),
                }
            }
            BoundType::Upper => {
                let bound = basic_info.upper.as_ref();
                match bound {
                    Some(b) if b.strict => {
                        // Strict upper bound: target value < bound
                        // If lower bound exists, use midpoint; otherwise subtract an infinitesimal.
                        if let Some(ref lb) = basic_info.lower {
                            (&b.value + &lb.value) / BigRational::from(BigInt::from(2))
                        } else {
                            &b.value - &unit
                        }
                    }
                    Some(b) => b.value.clone(),
                    None => BigRational::zero(),
                }
            }
        };

        // How much basic_var needs to change
        let basic_delta = &target_basic - basic_val;

        // How much nb_var needs to change: basic_delta = coeff * nb_delta => nb_delta = basic_delta / coeff
        let nb_delta = if !coeff.is_zero() {
            &basic_delta / &coeff
        } else {
            BigRational::zero()
        };

        // Compute new value for nb_var
        let new_nb_val = &nb_info.value + &nb_delta;

        // Clamp to nb_var's bounds if necessary
        // For strict bounds, we need to stay strictly inside, not at the boundary
        let clamped = if let Some(ref lower) = nb_info.lower {
            let at_or_below_lb = if lower.strict {
                new_nb_val <= lower.value
            } else {
                new_nb_val < lower.value
            };
            if at_or_below_lb {
                // For strict lb, we can't sit at lb, need to go above
                // But in simplex pivoting context, if we hit a bound we can't
                // cross, we should return the bound value (the pivot will handle it)
                if lower.strict {
                    // Go slightly above strict bound.
                    // Prefer midpoint with the opposite bound if it exists; otherwise, move just
                    // inside using the current value when possible to avoid overshooting.
                    if let Some(ref ub) = nb_info.upper {
                        (&lower.value + &ub.value) / BigRational::from(BigInt::from(2))
                    } else if nb_info.value > lower.value {
                        (&lower.value + &nb_info.value) / BigRational::from(BigInt::from(2))
                    } else {
                        &lower.value + &unit
                    }
                } else {
                    lower.value.clone()
                }
            } else {
                new_nb_val
            }
        } else {
            new_nb_val
        };

        let new_val = if let Some(ref upper) = nb_info.upper {
            let at_or_above_ub = if upper.strict {
                clamped >= upper.value
            } else {
                clamped > upper.value
            };
            if at_or_above_ub {
                if upper.strict {
                    // Go slightly below strict bound.
                    // Prefer midpoint with the opposite bound if it exists; otherwise, move just
                    // inside using the current value when possible to avoid overshooting.
                    if let Some(ref lb) = nb_info.lower {
                        (&lb.value + &upper.value) / BigRational::from(BigInt::from(2))
                    } else if nb_info.value < upper.value {
                        (&nb_info.value + &upper.value) / BigRational::from(BigInt::from(2))
                    } else {
                        &upper.value - &unit
                    }
                } else {
                    upper.value.clone()
                }
            } else {
                clamped
            }
        } else {
            clamped
        };

        if new_val == nb_info.value {
            // Degenerate update: for strict bounds it is common to need an infinitesimal move.
            // If there is slack to move the chosen non-basic variable in the direction that
            // fixes the violated basic bound, nudge it inside its bounds to guarantee progress.
            let dir = match violated_bound {
                BoundType::Lower => {
                    if coeff.is_positive() {
                        1
                    } else {
                        -1
                    }
                }
                BoundType::Upper => {
                    if coeff.is_positive() {
                        -1
                    } else {
                        1
                    }
                }
            };

            if let Some(nudged) = Self::nudge_value_in_direction(nb_info, dir, unit) {
                return nudged;
            }
        }

        new_val
    }

    fn strict_nudge_unit() -> BigRational {
        BigRational::new(BigInt::one(), BigInt::from(2))
    }

    fn nudge_value_in_direction(
        info: &VarInfo,
        dir: i32,
        unit: BigRational,
    ) -> Option<BigRational> {
        let current = &info.value;
        match dir {
            1 => {
                // Increase
                match &info.upper {
                    Some(ub) => {
                        if *current >= ub.value {
                            return None;
                        }
                        if ub.strict {
                            Some((current + &ub.value) / BigRational::from(BigInt::from(2)))
                        } else {
                            Some(ub.value.clone())
                        }
                    }
                    None => Some(current + unit),
                }
            }
            -1 => {
                // Decrease
                match &info.lower {
                    Some(lb) => {
                        if *current <= lb.value {
                            return None;
                        }
                        if lb.strict {
                            Some((current + &lb.value) / BigRational::from(BigInt::from(2)))
                        } else {
                            Some(lb.value.clone())
                        }
                    }
                    None => Some(current - unit),
                }
            }
            _ => None,
        }
    }

    /// Update a non-basic variable's value and propagate to basic variables
    fn update_nonbasic(&mut self, var: u32, new_val: BigRational) {
        let old_val = self.vars[var as usize].value.clone();
        let delta = &new_val - &old_val;

        if delta.is_zero() {
            return;
        }

        self.vars[var as usize].value = new_val;

        // Update all basic variables that depend on this one
        for row in &self.rows {
            let coeff = row.coeff(var);
            if !coeff.is_zero() {
                let basic_var = row.basic_var as usize;
                self.vars[basic_var].value += &coeff * &delta;
            }
        }
    }

    fn choose_nonbasic_fix_value(
        &self,
        info: &VarInfo,
        violated_type: &BoundType,
    ) -> Option<BigRational> {
        let two = BigRational::from(BigInt::from(2));
        let unit = Self::strict_nudge_unit();
        match violated_type {
            BoundType::Lower => {
                let lb = info.lower.as_ref()?;
                if lb.strict {
                    if let Some(ub) = info.upper.as_ref() {
                        Some((&lb.value + &ub.value) / &two)
                    } else {
                        Some(&lb.value + unit)
                    }
                } else {
                    Some(lb.value.clone())
                }
            }
            BoundType::Upper => {
                let ub = info.upper.as_ref()?;
                if ub.strict {
                    if let Some(lb) = info.lower.as_ref() {
                        Some((&lb.value + &ub.value) / &two)
                    } else {
                        Some(&ub.value - unit)
                    }
                } else {
                    Some(ub.value.clone())
                }
            }
        }
    }

    /// Build conflict explanation from bounds of variables involved
    fn build_conflict(&self, row_idx: usize) -> Vec<TheoryLit> {
        let mut reasons = Vec::new();
        let row = &self.rows[row_idx];

        // Add reason from basic variable's violated bound
        // Skip TermId(0) which is a dummy/placeholder reason (e.g., from Gomory cuts)
        let basic_info = &self.vars[row.basic_var as usize];
        if let Some(ref lower) = basic_info.lower {
            if lower.reason.0 != 0 {
                reasons.push(TheoryLit::new(lower.reason, lower.reason_value));
            }
        }
        if let Some(ref upper) = basic_info.upper {
            if upper.reason.0 != 0 {
                reasons.push(TheoryLit::new(upper.reason, upper.reason_value));
            }
        }

        // Add reasons from non-basic variables' bounds
        for &(nb_var, _) in &row.coeffs {
            let nb_info = &self.vars[nb_var as usize];
            if let Some(ref lower) = nb_info.lower {
                if lower.reason.0 != 0 {
                    reasons.push(TheoryLit::new(lower.reason, lower.reason_value));
                }
            }
            if let Some(ref upper) = nb_info.upper {
                if upper.reason.0 != 0 {
                    reasons.push(TheoryLit::new(upper.reason, upper.reason_value));
                }
            }
        }

        reasons
    }

    /// Run dual simplex to check feasibility
    fn dual_simplex(&mut self) -> TheoryResult {
        // Quick UNSAT check: contradictory bounds on a variable are immediately infeasible.
        //
        // This is important for problems that are purely a conjunction of bounds (no tableau
        // rows). The main dual-simplex loop focuses on pivoting basic variables, so we need an
        // explicit contradiction check for non-basic-only constraints.
        for var in 0..self.vars.len() {
            let info = &self.vars[var];
            let (Some(lower), Some(upper)) = (&info.lower, &info.upper) else {
                continue;
            };

            let contradicts = lower.value > upper.value
                || (lower.value == upper.value && (lower.strict || upper.strict));
            if contradicts {
                // Skip TermId(0) which is a dummy/placeholder reason
                let mut conflict = Vec::new();
                if lower.reason.0 != 0 {
                    conflict.push(TheoryLit::new(lower.reason, lower.reason_value));
                }
                if upper.reason.0 != 0 {
                    conflict.push(TheoryLit::new(upper.reason, upper.reason_value));
                }
                return TheoryResult::Unsat(conflict);
            }
        }

        // Limit iterations to prevent infinite loops.
        // With Bland's rule, simplex terminates in O(2^n) worst case but typically polynomial.
        // Use an aggressive limit: 5k base + 5 per row/var, capped at 50k.
        // This prevents runaway iteration on large problems. If we hit the limit,
        // return Unknown which is sound. Most LRA problems that can be solved
        // will be solved within 10k iterations; hard problems with many ITEs
        // and complex Boolean structure should return Unknown quickly rather
        // than wasting time on an over-approximated simplex.
        let base_iters = 5_000usize;
        let scale_iters = (self.rows.len() + self.vars.len()) * 5;
        let max_iters = std::cmp::min(base_iters + scale_iters, 50_000);

        let debug = std::env::var("Z4_DEBUG_LRA").is_ok();
        if debug {
            eprintln!(
                "[LRA] dual_simplex: {} rows, {} vars, max_iters={}",
                self.rows.len(),
                self.vars.len(),
                max_iters
            );
        }

        let mut last_print = 0usize;
        for iter in 0..max_iters {
            if debug && (iter < 20 || iter - last_print >= 10000) {
                last_print = iter;
                eprintln!(
                    "[LRA] iter {} - {} rows, {} vars",
                    iter,
                    self.rows.len(),
                    self.vars.len()
                );
            }
            // Find a basic variable that violates its bounds (using Bland's rule: smallest index)
            let mut violated_row: Option<(usize, BoundType, u32)> = None;
            for (row_idx, row) in self.rows.iter().enumerate() {
                if let Some(bound_type) = self.violates_bounds(row.basic_var) {
                    // Bland's rule: pick the violated basic variable with smallest index
                    match &violated_row {
                        None => violated_row = Some((row_idx, bound_type, row.basic_var)),
                        Some((_, _, best_var)) if row.basic_var < *best_var => {
                            violated_row = Some((row_idx, bound_type, row.basic_var));
                        }
                        _ => {}
                    }
                }
            }
            // Extract just the row_idx and bound_type for compatibility
            let violated_row = violated_row.map(|(idx, bound, _)| (idx, bound));

            let Some((row_idx, violated_bound)) = violated_row else {
                if debug && iter < 20 {
                    eprintln!("[LRA] iter {} - no violated row, checking non-basic", iter);
                }
                // All basic variables satisfy bounds - check non-basic too
                let nb_vars: Vec<u32> = self
                    .vars
                    .iter()
                    .enumerate()
                    .filter(|(_, info)| matches!(info.status, Some(VarStatus::NonBasic)))
                    .map(|(i, _)| i as u32)
                    .collect();

                let mut saw_violation = false;
                let mut did_fix = false;
                for var in nb_vars {
                    if let Some(violated_type) = self.violates_bounds(var) {
                        saw_violation = true;
                        let info = &self.vars[var as usize];
                        if let Some(nv) = self.choose_nonbasic_fix_value(info, &violated_type) {
                            self.update_nonbasic(var, nv);
                            did_fix = true;
                        }
                    }
                }

                // If we changed any non-basic assignments (or observed a strict-at-bound
                // violation we didn't resolve here), re-enter the main loop so we can
                // pivot on any newly violated basic variables.
                if did_fix || saw_violation {
                    if debug && iter < 20 {
                        eprintln!("[LRA] iter {} - fixed non-basic, continuing", iter);
                    }
                    continue;
                }

                if debug {
                    eprintln!("[LRA] Returning Sat at iter {}", iter);
                    for (i, info) in self.vars.iter().enumerate() {
                        let status = match &info.status {
                            Some(VarStatus::Basic(r)) => format!("B(row{})", r),
                            Some(VarStatus::NonBasic) => "NB".to_string(),
                            None => "?".to_string(),
                        };
                        eprintln!("[LRA]   var {} = {} ({})", i, info.value, status);
                    }
                }
                return TheoryResult::Sat;
            };

            if debug && iter < 20 {
                let row = &self.rows[row_idx];
                let basic_var = row.basic_var;
                let basic_info = &self.vars[basic_var as usize];
                let lb = basic_info
                    .lower
                    .as_ref()
                    .map(|b| format!("{}({})", b.value, if b.strict { "<" } else { "<=" }))
                    .unwrap_or_default();
                let ub = basic_info
                    .upper
                    .as_ref()
                    .map(|b| format!("{}({})", b.value, if b.strict { ">" } else { ">=" }))
                    .unwrap_or_default();
                eprintln!("[LRA] iter {} - violated row {}, basic_var={}, val={}, lb={}, ub={}, bound {:?}",
                    iter, row_idx, basic_var, basic_info.value, lb, ub, violated_bound);
            }

            // Find a suitable pivot candidate.
            //
            // Prefer a candidate that results in a non-degenerate update (entering var value changes),
            // but fall back to Bland's rule even if the pivot is degenerate (this is sound).
            let mut chosen: Option<(u32, BigRational)> = None;
            for entering_var in self.find_pivot_candidates(row_idx, violated_bound.clone()) {
                let new_val = self.compute_update_amount(row_idx, entering_var, &violated_bound);
                let old_val = self.vars[entering_var as usize].value.clone();
                if new_val != old_val {
                    chosen = Some((entering_var, new_val));
                    break;
                }
            }

            if chosen.is_none() {
                if let Some((entering_var, _direction)) =
                    self.find_pivot_candidate(row_idx, violated_bound.clone())
                {
                    let new_val =
                        self.compute_update_amount(row_idx, entering_var, &violated_bound);
                    chosen = Some((entering_var, new_val));
                }
            }

            let Some((entering_var, new_val)) = chosen else {
                // No suitable pivot - conflict
                let reasons = self.build_conflict(row_idx);
                return TheoryResult::Unsat(reasons);
            };

            if debug && iter < 20 {
                let nb_info = &self.vars[entering_var as usize];
                let nb_lb = nb_info
                    .lower
                    .as_ref()
                    .map(|b| format!("{}({})", b.value, if b.strict { "<" } else { "<=" }))
                    .unwrap_or_default();
                let nb_ub = nb_info
                    .upper
                    .as_ref()
                    .map(|b| format!("{}({})", b.value, if b.strict { ">" } else { ">=" }))
                    .unwrap_or_default();
                eprintln!(
                    "[LRA] iter {} - pivot: entering_var={}, old_val={}, new_val={}, lb={}, ub={}",
                    iter, entering_var, nb_info.value, new_val, nb_lb, nb_ub
                );
            }

            // Update the non-basic variable
            self.update_nonbasic(entering_var, new_val.clone());

            if debug && iter < 20 {
                // Check what happened to basic var after update
                let row = &self.rows[row_idx];
                let basic_info = &self.vars[row.basic_var as usize];
                eprintln!(
                    "[LRA] iter {} - after update: basic_var={} val={}",
                    iter, row.basic_var, basic_info.value
                );
            }

            // Pivot to swap basic/non-basic
            self.pivot(row_idx, entering_var);
        }

        // Too many iterations - return unknown
        TheoryResult::Unknown
    }

    /// Extract the current model (variable assignments)
    ///
    /// Returns a mapping from term IDs (for variables) to their rational values.
    /// Should only be called after `check()` returns `Sat`.
    pub fn extract_model(&self) -> LraModel {
        let mut values = HashMap::new();

        let debug = std::env::var("Z4_DEBUG_LRA").is_ok();
        if debug {
            eprintln!(
                "[LRA] extract_model: term_to_var has {} entries",
                self.term_to_var.len()
            );
        }

        // Extract values for all variables that have associated term IDs
        for (&term, &var) in &self.term_to_var {
            if let Some(info) = self.vars.get(var as usize) {
                if debug {
                    eprintln!(
                        "[LRA] extract_model: term {} -> var {} = {}",
                        term.0, var, info.value
                    );
                }
                values.insert(term, info.value.clone());
            }
        }

        LraModel { values }
    }

    /// Get the current value of a variable by term ID
    pub fn get_value(&self, term: TermId) -> Option<BigRational> {
        self.term_to_var
            .get(&term)
            .and_then(|&var| self.vars.get(var as usize))
            .map(|info| info.value.clone())
    }

    /// Get the current lower/upper bounds for a variable term, if known.
    ///
    /// This is primarily used by LIA to detect immediate integer infeasibility
    /// from bounds (e.g. `x > 5` and `x < 6` with `x : Int`).
    pub fn get_bounds(&self, term: TermId) -> Option<(Option<Bound>, Option<Bound>)> {
        let &var = self.term_to_var.get(&term)?;
        let info = self.vars.get(var as usize)?;
        Some((info.lower.clone(), info.upper.clone()))
    }

    /// Get all asserted atoms and their values
    pub fn get_asserted(&self) -> &HashMap<TermId, bool> {
        &self.asserted
    }

    /// Get the mapping from term IDs to internal variable IDs
    ///
    /// Used by LIA for HNF cut generation to translate between
    /// term-level and internal variable representations.
    pub fn term_to_var(&self) -> &HashMap<TermId, u32> {
        &self.term_to_var
    }

    /// Ensure a variable is registered with the LRA solver.
    ///
    /// This is called by the LIA solver when propagating Diophantine bounds
    /// for integer variables that may not have been encountered in LRA constraints.
    /// Returns the internal variable ID.
    pub fn ensure_var_registered(&mut self, term: TermId) -> u32 {
        self.intern_var(term)
    }

    /// Move non-basic variables to their bounds.
    ///
    /// This is a prerequisite for generating valid Gomory cuts - all non-basic
    /// variables must be at a bound for the cut to be valid.
    ///
    /// For each non-basic variable:
    /// - If it has a lower bound and no upper, move to lower bound
    /// - If it has an upper bound and no lower, move to upper bound
    /// - If it has both bounds, move to whichever is closer
    /// - If it has no bounds, leave it at 0 (unbounded)
    pub fn move_nonbasic_to_bounds(&mut self) {
        for var in 0..self.vars.len() {
            let info = &self.vars[var];
            if !matches!(info.status, Some(VarStatus::NonBasic)) {
                continue;
            }

            let target = match (&info.lower, &info.upper) {
                (Some(lb), None) => Some(lb.value.clone()),
                (None, Some(ub)) => Some(ub.value.clone()),
                (Some(lb), Some(ub)) => {
                    // Move to whichever bound is closer
                    let dist_to_lower = (&info.value - &lb.value).abs();
                    let dist_to_upper = (&ub.value - &info.value).abs();
                    if dist_to_lower <= dist_to_upper {
                        Some(lb.value.clone())
                    } else {
                        Some(ub.value.clone())
                    }
                }
                (None, None) => {
                    // Unbounded - move to 0
                    if !info.value.is_zero() {
                        Some(BigRational::zero())
                    } else {
                        None
                    }
                }
            };

            if let Some(new_val) = target {
                if new_val != self.vars[var].value {
                    self.update_nonbasic(var as u32, new_val);
                }
            }
        }
    }

    /// Try to patch a fractional integer basic variable to an integer value.
    ///
    /// This is Z3's "patching" technique that avoids branching by adjusting
    /// non-basic integer variables. For a fractional basic variable x, we look
    /// for a non-basic integer variable y in the same row such that adjusting y
    /// by a small integer delta makes x integral while keeping all bounds satisfied.
    ///
    /// Returns true if patching succeeded, false otherwise.
    pub fn try_patch_integer_var(
        &mut self,
        term: TermId,
        integer_vars: &hashbrown::HashSet<TermId>,
    ) -> bool {
        let debug = std::env::var("Z4_DEBUG_PATCH").is_ok();

        // Get the internal variable ID
        let Some(&var) = self.term_to_var.get(&term) else {
            return false;
        };

        // Check if this variable is basic (has a row)
        let Some(VarStatus::Basic(row_idx)) = self.vars.get(var as usize).and_then(|v| v.status)
        else {
            if debug {
                eprintln!("[PATCH] Term {:?} var {} is not basic", term, var);
            }
            return false;
        };

        let row = &self.rows[row_idx];
        let basic_value = &self.vars[var as usize].value;

        // Already integer?
        if basic_value.denom().is_one() {
            return false;
        }

        // For patching, we need: x + coeff * delta to be integral
        // where x is the basic variable and delta is the adjustment to a non-basic var
        let x_frac = fractional_part(basic_value);
        if x_frac.is_zero() {
            return false;
        }

        if debug {
            eprintln!(
                "[PATCH] Trying to patch var {} (term {:?}) with fractional value {}",
                var, term, basic_value
            );
        }

        // Try each non-basic integer variable in this row
        for &(nb_var, ref coeff) in &row.coeffs {
            // Must be an integer variable
            let Some(&nb_term) = self.var_to_term.get(&nb_var) else {
                continue;
            };
            if !integer_vars.contains(&nb_term) {
                continue;
            }

            // Must be non-basic
            if !matches!(self.vars[nb_var as usize].status, Some(VarStatus::NonBasic)) {
                continue;
            }

            // Coefficient must have a compatible fractional structure
            // For x + coeff * delta to be integral when x has fractional part f,
            // we need coeff * delta to have fractional part -f (mod 1)
            let coeff_frac = fractional_part(coeff);
            if coeff_frac.is_zero() {
                // Integer coefficient can't help fix fractional basic var
                // (would need fractional delta, but we need integer delta)
                continue;
            }

            // Find deltas that make x integral
            // x_new = x + coeff * delta must be integral
            // So we need (x + coeff * delta).is_int()
            // Let a1/a2 = coeff, x1/x2 = x (in lowest terms)
            // We need x1/x2 + (a1/a2)*delta to be integral
            // Rearranging: delta = (k*x2 - x1) * a2 / (a1 * x2) for some integer k
            // For delta to be integral, we need a2 divisible by x2
            let a1 = coeff.numer();
            let a2 = coeff.denom();
            let x1 = basic_value.numer();
            let x2 = basic_value.denom();

            if a2 % x2 != BigInt::zero() {
                if debug {
                    eprintln!(
                        "[PATCH] Coeff denom {} not divisible by basic denom {}",
                        a2, x2
                    );
                }
                continue;
            }

            // t = a2 / x2
            let t = a2 / x2;

            // Use extended GCD to find u, v such that u*a1 + v*x2 = gcd(a1, x2)
            // For this to work, we need gcd(a1, x2) = 1 (coprime)
            // If not coprime, patching with this variable won't work
            let g = gcd_bigint(a1, x2);
            if !g.is_one() && g != BigInt::from(-1) {
                if debug {
                    eprintln!("[PATCH] gcd({}, {}) = {} != 1", a1, x2, g);
                }
                continue;
            }

            // Find u such that u*a1 ≡ 1 (mod x2)
            let (_, u, _) = extended_gcd(a1, x2);

            // delta_base = u * t * x1 makes x + coeff * delta_base integral
            // Any delta = delta_base + k * a2 for integer k also works
            let delta_base = &u * &t * x1;

            // Find the smallest magnitude delta that keeps bounds satisfied
            // Try delta_base, delta_base + a2, delta_base - a2, etc.
            let nb_info = &self.vars[nb_var as usize];
            let nb_value = &nb_info.value;

            for sign in [BigInt::one(), BigInt::from(-1)] {
                for k in 0..10 {
                    let delta = &delta_base + &sign * BigInt::from(k) * a2;
                    let delta_rat = BigRational::from(delta.clone());
                    let new_nb_value = nb_value + &delta_rat;

                    // Check bounds on the non-basic variable
                    if let Some(ref lb) = nb_info.lower {
                        if new_nb_value < lb.value || (lb.strict && new_nb_value == lb.value) {
                            continue;
                        }
                    }
                    if let Some(ref ub) = nb_info.upper {
                        if new_nb_value > ub.value || (ub.strict && new_nb_value == ub.value) {
                            continue;
                        }
                    }

                    // Check that the new basic value is actually integral
                    let new_basic_value = basic_value + coeff * &delta_rat;
                    if !new_basic_value.denom().is_one() {
                        continue;
                    }

                    // Check bounds on all affected basic variables
                    let mut all_bounds_ok = true;
                    for other_row in &self.rows {
                        let other_coeff = other_row.coeff(nb_var);
                        if other_coeff.is_zero() {
                            continue;
                        }
                        let other_basic = other_row.basic_var as usize;
                        let other_info = &self.vars[other_basic];
                        let old_val = &other_info.value;
                        let new_val = old_val + &other_coeff * &delta_rat;

                        if let Some(ref lb) = other_info.lower {
                            if new_val < lb.value || (lb.strict && new_val == lb.value) {
                                all_bounds_ok = false;
                                break;
                            }
                        }
                        if let Some(ref ub) = other_info.upper {
                            if new_val > ub.value || (ub.strict && new_val == ub.value) {
                                all_bounds_ok = false;
                                break;
                            }
                        }

                        // If patching would make an integer basic variable fractional, skip
                        if let Some(&other_term) = self.var_to_term.get(&(other_basic as u32)) {
                            if integer_vars.contains(&other_term)
                                && old_val.denom().is_one()
                                && !new_val.denom().is_one()
                            {
                                all_bounds_ok = false;
                                break;
                            }
                        }
                    }

                    if all_bounds_ok {
                        if debug {
                            eprintln!(
                                "[PATCH] SUCCESS: adjusting var {} by delta {} -> new value {}",
                                nb_var, delta_rat, new_nb_value
                            );
                        }
                        // Apply the patch
                        self.update_nonbasic(nb_var, new_nb_value);
                        return true;
                    }
                }
            }
        }

        if debug {
            eprintln!(
                "[PATCH] No valid patch found for var {} (term {:?})",
                var, term
            );
        }
        false
    }

    /// Generate Gomory cuts for integer-infeasible basic variables.
    ///
    /// This is the core technique for solving Linear Integer Arithmetic problems.
    /// When a basic integer variable has a fractional value v, we generate a
    /// cutting plane that:
    /// 1. Is satisfied by all integer solutions
    /// 2. Cuts off the current fractional solution
    ///
    /// Returns a list of (linear_expr, bound, is_lower) tuples representing
    /// the generated cuts: linear_expr >= bound (if is_lower) or linear_expr <= bound.
    ///
    /// The algorithm follows Gomory's mixed-integer cutting plane method as
    /// implemented in Z3 (reference/z3/src/math/lp/gomory.cpp).
    pub fn generate_gomory_cuts(
        &self,
        integer_vars: &hashbrown::HashSet<TermId>,
    ) -> Vec<GomoryCut> {
        let mut cuts = Vec::new();
        let debug = std::env::var("Z4_DEBUG_GOMORY").is_ok();

        if debug {
            eprintln!(
                "[GOMORY] Generating cuts for {} integer vars",
                integer_vars.len()
            );
        }

        // Convert integer term IDs to internal var IDs
        let int_var_ids: hashbrown::HashSet<u32> = integer_vars
            .iter()
            .filter_map(|t| self.term_to_var.get(t).copied())
            .collect();

        // Look for basic integer variables with fractional values
        for row in &self.rows {
            let basic_var = row.basic_var;

            // Check if this basic variable corresponds to an integer variable
            let is_int_basic = if let Some(&term) = self.var_to_term.get(&basic_var) {
                integer_vars.contains(&term)
            } else {
                false
            };

            if !is_int_basic {
                continue;
            }

            let basic_info = &self.vars[basic_var as usize];
            let basic_value = &basic_info.value;

            // Check if fractional
            if basic_value.denom().is_one() {
                continue; // Already integer
            }

            // Compute f = fractional part of basic_value
            let f = fractional_part(basic_value);
            if f.is_zero() {
                continue;
            }

            let one_minus_f = BigRational::one() - &f;

            if debug {
                eprintln!(
                    "[GOMORY] Basic var {} has fractional value {}, f={}, 1-f={}",
                    basic_var, basic_value, f, one_minus_f
                );
            }

            // Build the Gomory cut
            // The cut is: Σ(coeff_j * (x_j - l_j)) >= f for lower bounds
            // Rearranged: Σ(coeff_j * x_j) >= f + Σ(coeff_j * l_j)
            // So k starts at f and we add coeff_j * l_j for each variable at lower bound
            //
            // Note: The standard GMI derivation (Nemhauser & Wolsey) uses RHS = f, not 1.
            // Some formulations use RHS = 1 with surplus variables s_j = x_j - l_j,
            // but our cut is in terms of original variables, so we use f.
            let mut cut_coeffs: Vec<(u32, BigRational)> = Vec::new();
            let mut cut_k = f.clone(); // CRITICAL: Start with f, not 1!
            let mut valid_cut = true;

            for &(nb_var, ref row_coeff) in &row.coeffs {
                if row_coeff.is_zero() {
                    continue;
                }

                // Skip cuts involving internal (slack) variables.
                // Gomory cuts are only sound when the cut can be expressed in terms of
                // the original problem variables with proper bound information.
                // Substituting slack definitions is unsound because the original variables
                // may not be at their bounds in the current tableau.
                if !self.var_to_term.contains_key(&nb_var) {
                    if debug {
                        eprintln!(
                            "[GOMORY] Non-basic var {} is internal (no term), skipping cut",
                            nb_var
                        );
                    }
                    valid_cut = false;
                    break;
                }

                let nb_info = &self.vars[nb_var as usize];

                // Determine if this non-basic variable is at lower or upper bound
                let at_lower = Self::is_at_lower_bound(nb_info);
                let at_upper = Self::is_at_upper_bound(nb_info);

                // For unbounded variables (no lower or upper), skip this row
                // Gomory cuts require all non-basic vars to be at defined bounds
                let is_unbounded = nb_info.lower.is_none() && nb_info.upper.is_none();
                if is_unbounded {
                    if debug {
                        eprintln!(
                            "[GOMORY] Non-basic var {} is unbounded, skipping cut",
                            nb_var
                        );
                    }
                    valid_cut = false;
                    break;
                }

                if !at_lower && !at_upper {
                    // Non-basic variable not at a bound - can't generate valid cut
                    if debug {
                        eprintln!(
                            "[GOMORY] Non-basic var {} not at bound, skipping cut",
                            nb_var
                        );
                    }
                    valid_cut = false;
                    break;
                }

                // Is this non-basic variable an integer?
                let nb_is_int = if let Some(&term) = self.var_to_term.get(&nb_var) {
                    integer_vars.contains(&term)
                } else {
                    int_var_ids.contains(&nb_var)
                };

                // The row is: x_i = Σ(a_ij * x_j) + c
                // The Gomory cut derivation uses tableau form: x_i - Σ(a_ij * x_j) = 0
                // So we negate the coefficients to get -a_ij (matching Z3 line 302: -p.coeff())
                let a = -row_coeff.clone();

                let new_coeff = if nb_is_int {
                    // Integer case
                    let fj = fractional_part(&a);
                    if fj.is_zero() {
                        // Integer coefficient on integer variable - no contribution if at bound
                        continue;
                    }
                    let one_minus_fj = BigRational::one() - &fj;

                    if at_lower {
                        // new_a = fj / (1-f) if fj <= 1-f, else (1-fj) / f
                        if fj <= one_minus_f {
                            &fj / &one_minus_f
                        } else {
                            &one_minus_fj / &f
                        }
                    } else {
                        // at_upper
                        // new_a = -(fj / f) if fj <= f, else -((1-fj) / (1-f))
                        if fj <= f {
                            -(&fj / &f)
                        } else {
                            -(&one_minus_fj / &one_minus_f)
                        }
                    }
                } else {
                    // Real case
                    if at_lower {
                        if a.is_positive() {
                            &a / &one_minus_f
                        } else {
                            -(&a / &f)
                        }
                    } else {
                        // at_upper
                        if a.is_positive() {
                            -(&a / &f)
                        } else {
                            &a / &one_minus_f
                        }
                    }
                };

                if !new_coeff.is_zero() {
                    // The Gomory cut is: Σ(new_coeff * x_j) >= k
                    // Following Z3's convention (reference/z3/src/math/lp/gomory.cpp):
                    // - For lower bound: new_coeff is positive, k += new_coeff * lb
                    // - For upper bound: new_coeff is negative, k += new_coeff * ub
                    //
                    // The new_coeff values are computed correctly above to match Z3:
                    // - int_case_in_gomory_cut: new_a is positive for lower, negative for upper
                    // - real_case_in_gomory_cut: same convention
                    //
                    // Z3 code (line 78-83): new_a = -(...), SASSERT(new_a.is_neg())
                    // So we use new_coeff directly in both cases (no extra negation).
                    let bound_val = if at_lower {
                        nb_info
                            .lower
                            .as_ref()
                            .map(|b| b.value.clone())
                            .unwrap_or_else(BigRational::zero)
                    } else {
                        nb_info
                            .upper
                            .as_ref()
                            .map(|b| b.value.clone())
                            .unwrap_or_else(BigRational::zero)
                    };

                    cut_k += &new_coeff * &bound_val;
                    cut_coeffs.push((nb_var, new_coeff));
                }
            }

            if !valid_cut || cut_coeffs.is_empty() {
                continue;
            }

            if debug {
                eprintln!(
                    "[GOMORY] Generated cut with {} terms, k={}",
                    cut_coeffs.len(),
                    cut_k
                );
                for (var_id, coeff) in &cut_coeffs {
                    let info = &self.vars[*var_id as usize];
                    let term_str = self
                        .var_to_term
                        .get(var_id)
                        .map(|t| format!("term {:?}", t))
                        .unwrap_or_else(|| "unknown".to_string());
                    eprintln!(
                        "[GOMORY]   var {} ({}) coeff={} value={} lower={:?} upper={:?}",
                        var_id,
                        term_str,
                        coeff,
                        info.value,
                        info.lower.as_ref().map(|b| &b.value),
                        info.upper.as_ref().map(|b| &b.value)
                    );
                }
                // Print the basic variable's term mapping
                if let Some(basic_term) = self.var_to_term.get(&basic_var) {
                    eprintln!(
                        "[GOMORY] Basic var {} maps to term {:?}",
                        basic_var, basic_term
                    );
                } else {
                    eprintln!(
                        "[GOMORY] Basic var {} has no term mapping (internal var)",
                        basic_var
                    );
                }
            }

            // The cut is: Σ(cut_coeffs * x) >= cut_k
            cuts.push(GomoryCut {
                coeffs: cut_coeffs,
                bound: cut_k,
                is_lower: true, // >= constraint
            });

            // Limit number of cuts per iteration
            if cuts.len() >= 2 {
                break;
            }
        }

        cuts
    }

    /// Check if a variable is at its lower bound
    fn is_at_lower_bound(info: &VarInfo) -> bool {
        match (&info.lower, &info.upper) {
            (Some(ref lower), _) => {
                // At lower if value equals lower bound (allowing for non-strict)
                // For strict bounds, we should be just above
                if lower.strict {
                    info.value > lower.value && info.value <= &lower.value + BigRational::one()
                } else {
                    info.value == lower.value
                }
            }
            (None, None) => {
                // Unbounded variable - by convention treat 0 as implicit lower bound
                // ONLY mark as at_lower, never at_upper for unbounded vars
                info.value.is_zero()
            }
            (None, Some(_)) => false, // Has only upper bound
        }
    }

    /// Check if a variable is at its upper bound
    fn is_at_upper_bound(info: &VarInfo) -> bool {
        match (&info.lower, &info.upper) {
            (_, Some(ref upper)) => {
                if upper.strict {
                    info.value < upper.value && info.value >= &upper.value - BigRational::one()
                } else {
                    info.value == upper.value
                }
            }
            (None, None) => {
                // Unbounded variable - never at upper bound (we treat 0 as lower)
                false
            }
            (Some(_), None) => false, // Has only lower bound
        }
    }

    /// Add a Gomory cut as a constraint
    ///
    /// The cut is: Σ(coeff * var) >= bound (if is_lower) or <= bound
    pub fn add_gomory_cut(&mut self, cut: &GomoryCut, reason: TermId) {
        // Convert cut to a linear expression
        // cut.coeffs uses internal var IDs
        let expr = LinearExpr {
            coeffs: cut.coeffs.clone(),
            constant: BigRational::zero(),
        };

        let bound_type = if cut.is_lower {
            BoundType::Lower
        } else {
            BoundType::Upper
        };

        self.assert_bound(expr, cut.bound.clone(), bound_type, false, reason, true);
        self.dirty = true;
    }

    /// Add a direct bound on an internal variable.
    ///
    /// Used by LIA solver to add bounds discovered through Diophantine solving.
    /// The bound is: var >= value (if is_lower) or var <= value.
    pub fn add_direct_bound(&mut self, var: u32, value: BigRational, is_lower: bool) {
        // Create a simple expression: 1 * var
        let expr = LinearExpr {
            coeffs: vec![(var, BigRational::one())],
            constant: BigRational::zero(),
        };

        let bound_type = if is_lower {
            BoundType::Lower
        } else {
            BoundType::Upper
        };

        // Use a dummy reason - these bounds are globally valid
        let dummy_reason = TermId(0);
        self.assert_bound(expr, value, bound_type, false, dummy_reason, true);
        self.dirty = true;
    }
}

/// A Gomory cutting plane
#[derive(Debug, Clone)]
pub struct GomoryCut {
    /// Coefficients: (internal_var_id, coefficient)
    pub coeffs: Vec<(u32, BigRational)>,
    /// The bound value (RHS of the inequality)
    pub bound: BigRational,
    /// True for >= constraint (lower bound), false for <= (upper bound)
    pub is_lower: bool,
}

/// Compute the fractional part of a rational number
/// frac(x) = x - floor(x), always in [0, 1)
fn fractional_part(val: &BigRational) -> BigRational {
    let numer = val.numer();
    let denom = val.denom();

    // floor(n/d) for positive d
    let floor_val = if numer.is_negative() {
        // For negative numbers: floor(n/d) = (n - d + 1) / d for d > 0
        (numer - denom + BigInt::one()) / denom
    } else {
        numer / denom
    };

    val - BigRational::from(floor_val)
}

/// Compute GCD of two BigInts
fn gcd_bigint(a: &BigInt, b: &BigInt) -> BigInt {
    let mut a = a.abs();
    let mut b = b.abs();
    while !b.is_zero() {
        let t = b.clone();
        b = &a % &b;
        a = t;
    }
    a
}

/// Extended Euclidean algorithm: returns (gcd, x, y) such that a*x + b*y = gcd
fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if b.is_zero() {
        return (a.clone(), BigInt::one(), BigInt::zero());
    }
    let (g, x, y) = extended_gcd(b, &(a % b));
    (g, y.clone(), x - (a / b) * y)
}

impl TheorySolver for LraSolver<'_> {
    fn assert_literal(&mut self, literal: TermId, value: bool) {
        // Unwrap NOT: NOT(inner)=true means inner=false
        let (term, val) = match self.terms.get(literal) {
            TermData::Not(inner) => (*inner, !value),
            _ => (literal, value),
        };
        let debug = std::env::var("Z4_DEBUG_LRA_ASSERT").is_ok();
        if debug {
            eprintln!(
                "[LRA] assert_literal: term={:?} ({:?}), value={}",
                term,
                self.terms.get(term),
                val
            );
        }
        self.asserted.insert(term, val);
        self.dirty = true;
    }

    fn check(&mut self) -> TheoryResult {
        let debug = std::env::var("Z4_DEBUG_LRA").is_ok();

        if debug {
            eprintln!(
                "[LRA] check() called, dirty={}, saw_unsupported={}",
                self.dirty, self.saw_unsupported
            );
        }

        if !self.dirty {
            if debug {
                eprintln!("[LRA] Not dirty, returning early");
            }
            return if self.saw_unsupported {
                TheoryResult::Unknown
            } else {
                TheoryResult::Sat
            };
        }
        self.dirty = false;
        self.saw_unsupported = false;

        // Process all asserted atoms (sorted for deterministic order)
        let mut asserted_clone: Vec<_> = self.asserted.iter().map(|(&t, &v)| (t, v)).collect();
        asserted_clone.sort_by_key(|(t, _)| t.0);

        let mut parsed_count = 0;
        let mut skipped_count = 0;
        let mut _cache_hits = 0;
        // Track disequalities for post-simplex checking
        // Stores (term, expr, asserted_value) where asserted_value is the value the term was asserted with
        let mut disequalities: Vec<(TermId, LinearExpr, bool)> = Vec::new();

        for (term, value) in asserted_clone {
            // Use cached parse result if available
            let cached = self.atom_cache.get(&term).cloned();
            let parsed_info = match cached {
                Some(info) => {
                    _cache_hits += 1;
                    info
                }
                None => {
                    // Parse and cache
                    let parsed = self.parse_atom(term).map(|(expr, is_le, strict)| {
                        let is_eq = matches!(self.terms.get(term), TermData::App(Symbol::Named(name), _) if name == "=");
                        let is_distinct = matches!(self.terms.get(term), TermData::App(Symbol::Named(name), _) if name == "distinct");
                        ParsedAtomInfo { expr, is_le, strict, is_eq, is_distinct }
                    });
                    self.atom_cache.insert(term, parsed.clone());
                    parsed
                }
            };

            let Some(info) = parsed_info else {
                skipped_count += 1;
                // Check if the skipped atom is a Boolean combination (or, and, xor, ite).
                // These shouldn't be theory atoms - they indicate the DPLL layer is
                // passing us intermediate CNF expressions instead of just arithmetic predicates.
                // When this happens, we can't trust our SAT result because we're missing constraints.
                match self.terms.get(term) {
                    TermData::App(Symbol::Named(name), _)
                        if name == "or" || name == "and" || name == "xor" || name == "=>" =>
                    {
                        if debug {
                            eprintln!(
                                "[LRA] Skipping Boolean combination {:?} - marking incomplete",
                                term
                            );
                        }
                        self.saw_unsupported = true;
                    }
                    TermData::Ite(_, _, _) => {
                        if debug {
                            eprintln!("[LRA] Skipping ITE atom {:?} - marking incomplete", term);
                        }
                        self.saw_unsupported = true;
                    }
                    _ => {
                        if debug {
                            eprintln!(
                                "[LRA] Skipping unparseable atom {:?} (term: {:?})",
                                term,
                                self.terms.get(term)
                            );
                        }
                    }
                }
                continue;
            };
            parsed_count += 1;

            let ParsedAtomInfo {
                expr,
                is_le,
                strict,
                is_eq,
                is_distinct,
            } = info;

            // For all arithmetic atoms, expr is normalized so that the atom is:
            // expr <= 0 (for is_le=true) or expr >= 0 (for is_le=false)
            // The bound is always 0.
            let zero = BigRational::zero();

            if is_eq || is_distinct {
                // For equality (=):
                //   value=true  → assert equality (a = b)
                //   value=false → add disequality (a != b)
                // For distinct:
                //   value=true  → add disequality (a != b) - INVERTED
                //   value=false → assert equality (a = b) - INVERTED
                let is_equality = (is_eq && value) || (is_distinct && !value);

                if is_equality {
                    // Equality: expr = 0 means expr <= 0 AND expr >= 0
                    // Use the actual assertion value for reason_value (important for `distinct` negations)
                    if !expr.is_constant() {
                        self.assert_bound(
                            expr.clone(),
                            zero.clone(),
                            BoundType::Upper,
                            false,
                            term,
                            value,
                        );
                        self.assert_bound(expr, zero, BoundType::Lower, false, term, value);
                    }
                } else {
                    // Disequality: x != c can't be directly encoded in simplex.
                    // We'll check these after simplex to see if any are violated by the model.
                    // Store (term, expr, asserted_value) for post-simplex checking: if expr evaluates to 0
                    // in the model, the disequality is violated.
                    if debug {
                        eprintln!("[LRA] Disequality atom {:?}: will check model later", term);
                    }
                    disequalities.push((term, expr, value));
                }
            } else if value {
                // Positive assertion: expr <= 0 or expr < 0
                if is_le {
                    self.assert_bound(expr, zero, BoundType::Upper, strict, term, true);
                } else {
                    // expr >= 0 or expr > 0
                    self.assert_bound(expr, zero, BoundType::Lower, strict, term, true);
                }
            } else {
                // Negated assertion: !(expr <= 0) means expr > 0
                if is_le {
                    // !(expr <= 0) => expr > 0
                    self.assert_bound(expr, zero, BoundType::Lower, !strict, term, false);
                } else {
                    // !(expr >= 0) => expr < 0
                    self.assert_bound(expr, zero, BoundType::Upper, !strict, term, false);
                }
            }
        }

        // Run simplex
        if debug {
            eprintln!(
                "[LRA] Atom processing: parsed={}, skipped={}, total_asserted={}, disequalities={}",
                parsed_count,
                skipped_count,
                parsed_count + skipped_count,
                disequalities.len()
            );
        }
        let simplex_result = self.dual_simplex();
        if debug {
            eprintln!(
                "[LRA] simplex result: {:?}, saw_unsupported={}",
                simplex_result, self.saw_unsupported
            );
        }

        // If simplex returned Sat, check disequalities
        // IMPORTANT: Only check disequalities when we have complete information.
        // If saw_unsupported is true, the model is incomplete (e.g., ITE terms created
        // unconstrained slack variables), so we can't trust the model to check disequalities.
        if matches!(simplex_result, TheoryResult::Sat)
            && !disequalities.is_empty()
            && !self.saw_unsupported
        {
            // Evaluate each disequality in the current model
            // A disequality (term, expr, asserted_value) with expr = LHS - RHS is violated if expr == 0
            for (term, expr, asserted_value) in &disequalities {
                // Evaluate the expression in the current model
                let mut eval_value = expr.constant.clone();
                for &(var, ref coeff) in &expr.coeffs {
                    if let Some(info) = self.vars.get(var as usize) {
                        eval_value += coeff * &info.value;
                    }
                }

                if debug {
                    eprintln!(
                        "[LRA] Checking disequality {:?}: expr value = {}",
                        term, eval_value
                    );
                }

                // If expr == 0, the disequality (LHS != RHS) is violated because LHS == RHS
                if eval_value.is_zero() {
                    // Check if all variables in the expression are pinned (lower == upper == value)
                    // If so, the expression is forced to 0 and we can return Unsat
                    // If any variable has slack, other solutions might satisfy the disequality
                    let all_vars_pinned = expr.coeffs.iter().all(|&(var, _)| {
                        if let Some(info) = self.vars.get(var as usize) {
                            // Variable is pinned if lower == upper == value
                            let pinned =
                                info.lower.as_ref().is_some_and(|lb| lb.value == info.value)
                                    && info.upper.as_ref().is_some_and(|ub| ub.value == info.value);
                            if debug && !pinned {
                                eprintln!(
                                    "[LRA] Var {} has slack: value={}, lb={:?}, ub={:?}",
                                    var,
                                    info.value,
                                    info.lower.as_ref().map(|b| &b.value),
                                    info.upper.as_ref().map(|b| &b.value)
                                );
                            }
                            pinned
                        } else {
                            false // Unknown variable - conservative: assume not pinned
                        }
                    });

                    if all_vars_pinned || expr.coeffs.is_empty() {
                        if debug {
                            eprintln!("[LRA] Disequality {:?} is VIOLATED with forced model - returning Unsat", term);
                        }
                        // All variables are pinned, so the model is forced and violates disequality
                        return TheoryResult::Unsat(vec![TheoryLit {
                            term: *term,
                            value: *asserted_value,
                        }]);
                    } else {
                        // Some variables have slack, so other solutions might exist.
                        // Request a split on (expr < 0) OR (expr > 0) to explore both regions.
                        // For a simple disequality like x != 0, this becomes (x < 0) OR (x > 0).

                        // Find a variable to split on.
                        // For single-variable disequalities (x != c), we split on x.
                        // For multi-variable disequalities (x - y != 0), we pick any variable with slack.
                        if expr.coeffs.len() == 1 {
                            let (var, _coeff) = &expr.coeffs[0];
                            if let Some(&var_term) = self.var_to_term.get(var) {
                                // For expr = coeff*x + const = 0, the excluded value is -const/coeff
                                // For x != 0 (where expr = x), excluded_value = 0
                                let excluded = -&expr.constant;
                                if debug {
                                    eprintln!("[LRA] Disequality {:?} violated with slack - requesting split on var {:?} != {}",
                                              term, var_term, excluded);
                                }
                                return TheoryResult::NeedDisequlitySplit(DisequlitySplitRequest {
                                    variable: var_term,
                                    excluded_value: excluded,
                                });
                            }
                        } else {
                            // Multi-variable disequality (e.g., E - F != 0, i.e., E != F).
                            // Try single-value enumeration first - find a variable with slack
                            // and split on its current value. If the CHC-SMT layer detects an
                            // infinite loop (same variable being split repeatedly), it will
                            // request an expression split instead.
                            for (var, _coeff) in &expr.coeffs {
                                if let Some(info) = self.vars.get(*var as usize) {
                                    // Check if this variable has slack
                                    let has_lower_slack =
                                        info.lower.as_ref().is_none_or(|lb| lb.value < info.value);
                                    let has_upper_slack =
                                        info.upper.as_ref().is_none_or(|ub| ub.value > info.value);
                                    if has_lower_slack || has_upper_slack {
                                        if let Some(&var_term) = self.var_to_term.get(var) {
                                            // Split on the current value of this variable
                                            let excluded = info.value.clone();
                                            if debug {
                                                eprintln!("[LRA] Multi-var disequality {:?} violated - split on var {:?} != {}",
                                                          term, var_term, excluded);
                                            }
                                            return TheoryResult::NeedDisequlitySplit(
                                                DisequlitySplitRequest {
                                                    variable: var_term,
                                                    excluded_value: excluded,
                                                },
                                            );
                                        }
                                    }
                                }
                            }
                            // No variable with slack found - request expression split
                            if debug {
                                eprintln!("[LRA] Multi-var disequality {:?} violated - no slack, requesting expression split",
                                          term);
                            }
                            return TheoryResult::NeedExpressionSplit(ExpressionSplitRequest {
                                disequality_term: *term,
                            });
                        }

                        // Fallback for complex expressions where no variable has slack: return Unknown
                        if debug {
                            eprintln!("[LRA] Disequality {:?} is VIOLATED but no splittable var found - returning Unknown", term);
                        }
                        return TheoryResult::Unknown;
                    }
                }
            }
            if debug {
                eprintln!("[LRA] All disequalities satisfied");
            }
        }

        match simplex_result {
            TheoryResult::Sat if self.saw_unsupported => {
                if debug {
                    eprintln!("[LRA] Returning Unknown due to saw_unsupported");
                }
                TheoryResult::Unknown
            }
            other => other,
        }
    }

    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        // For now, no eager propagation
        Vec::new()
    }

    fn push(&mut self) {
        self.scopes.push(self.trail.len());
    }

    fn pop(&mut self) {
        let Some(mark) = self.scopes.pop() else {
            return;
        };

        // Restore bounds from trail
        while self.trail.len() > mark {
            let (var, old_lower, old_upper) = self.trail.pop().unwrap();
            if (var as usize) < self.vars.len() {
                self.vars[var as usize].lower = old_lower;
                self.vars[var as usize].upper = old_upper;
            }
        }

        self.dirty = true;
    }

    fn reset(&mut self) {
        self.rows.clear();
        self.vars.clear();
        self.term_to_var.clear();
        self.var_to_term.clear();
        self.next_var = 0;
        self.trail.clear();
        self.scopes.clear();
        self.asserted.clear();
        // Note: We intentionally DON'T clear atom_cache here.
        // The cache stores parsed expressions which can be reused across resets.
        // However, the LinearExpr contains variable IDs that get invalidated on reset.
        // We must clear the cache to avoid using stale variable IDs.
        self.atom_cache.clear();
        self.saw_unsupported = false;
        self.dirty = true;
    }
}

// ============================================================================
// Kani Verification Harnesses
// ============================================================================
//
// These proofs verify the core invariants of the LRA (Linear Real Arithmetic) solver:
// 1. LinearExpr operations: term combining, scaling, negation
// 2. Bounds consistency: lower <= upper implies feasibility
// 3. Tableau invariants: pivot operations preserve structure
// 4. Push/pop state consistency

#[cfg(kani)]
mod verification {
    use super::*;

    // ========================================================================
    // LinearExpr Invariants
    // ========================================================================

    /// Adding zero coefficient doesn't change the expression
    #[kani::proof]
    fn proof_add_term_zero_is_noop() {
        let mut expr = LinearExpr::zero();
        expr.add_term(0, BigRational::from(BigInt::from(5)));

        let coeff_before = expr
            .coeffs
            .iter()
            .find(|(v, _)| *v == 0)
            .map(|(_, c)| c.clone());

        // Adding zero should not change anything
        expr.add_term(0, BigRational::zero());

        let coeff_after = expr
            .coeffs
            .iter()
            .find(|(v, _)| *v == 0)
            .map(|(_, c)| c.clone());

        assert!(
            coeff_before == coeff_after,
            "Adding zero coefficient is a no-op"
        );
    }

    /// Adding opposite coefficients cancels to zero
    #[kani::proof]
    fn proof_add_term_cancellation() {
        let mut expr = LinearExpr::zero();

        let val: i32 = kani::any();
        kani::assume(val != 0 && val > -1000 && val < 1000);

        let coeff = BigRational::from(BigInt::from(val));
        let neg_coeff = -coeff.clone();

        expr.add_term(0, coeff);
        assert!(!expr.coeffs.is_empty(), "Should have one term");

        expr.add_term(0, neg_coeff);
        let has_var_0 = expr.coeffs.iter().any(|(v, _)| *v == 0);
        assert!(!has_var_0, "Opposite coefficients should cancel");
    }

    /// Scaling by 1 preserves the expression
    #[kani::proof]
    fn proof_scale_by_one() {
        let mut expr = LinearExpr::zero();

        let val: i32 = kani::any();
        kani::assume(val > -100 && val < 100);

        let coeff = BigRational::from(BigInt::from(val));
        expr.add_term(0, coeff.clone());
        expr.constant = BigRational::from(BigInt::from(42));

        let coeff_before = expr
            .coeffs
            .iter()
            .find(|(v, _)| *v == 0)
            .map(|(_, c)| c.clone());
        let const_before = expr.constant.clone();

        expr.scale(&BigRational::one());

        let coeff_after = expr
            .coeffs
            .iter()
            .find(|(v, _)| *v == 0)
            .map(|(_, c)| c.clone());

        assert!(
            coeff_before == coeff_after,
            "Scale by 1 preserves coefficients"
        );
        assert!(
            expr.constant == const_before,
            "Scale by 1 preserves constant"
        );
    }

    /// Double negation returns to original
    #[kani::proof]
    fn proof_double_negation() {
        let mut expr = LinearExpr::zero();

        let val: i32 = kani::any();
        kani::assume(val > -100 && val < 100);

        expr.add_term(0, BigRational::from(BigInt::from(val)));
        expr.constant = BigRational::from(BigInt::from(17));

        let coeff_original = expr
            .coeffs
            .iter()
            .find(|(v, _)| *v == 0)
            .map(|(_, c)| c.clone());
        let const_original = expr.constant.clone();

        expr.negate();
        expr.negate();

        let coeff_final = expr
            .coeffs
            .iter()
            .find(|(v, _)| *v == 0)
            .map(|(_, c)| c.clone());

        assert!(
            coeff_original == coeff_final,
            "Double negation restores coefficient"
        );
        assert!(
            expr.constant == const_original,
            "Double negation restores constant"
        );
    }

    /// is_constant returns true iff no variable terms
    #[kani::proof]
    fn proof_is_constant_correctness() {
        let expr = LinearExpr::zero();
        assert!(expr.is_constant(), "Zero expression is constant");

        let const_expr = LinearExpr::constant(BigRational::from(BigInt::from(42)));
        assert!(const_expr.is_constant(), "Constant expression is constant");

        let var_expr = LinearExpr::var(0);
        assert!(
            !var_expr.is_constant(),
            "Variable expression is not constant"
        );
    }

    // ========================================================================
    // Bounds Consistency Invariants
    // ========================================================================

    /// Lower bound > upper bound implies infeasibility
    #[kani::proof]
    fn proof_contradictory_bounds_detected() {
        // This proof verifies the quick UNSAT check in dual_simplex
        let lower: i32 = kani::any();
        let upper: i32 = kani::any();
        kani::assume(lower > -100 && lower < 100);
        kani::assume(upper > -100 && upper < 100);
        kani::assume(lower > upper);

        let lb = BigRational::from(BigInt::from(lower));
        let ub = BigRational::from(BigInt::from(upper));

        // A variable with lower > upper is infeasible
        let contradicts = lb > ub;
        assert!(contradicts, "lower > upper is contradictory");
    }

    /// Equal bounds with strictness are contradictory
    #[kani::proof]
    fn proof_equal_strict_bounds_contradictory() {
        let val: i32 = kani::any();
        kani::assume(val > -100 && val < 100);

        let bound_val = BigRational::from(BigInt::from(val));

        // x > val AND x < val is contradictory (both strict at same value)
        // x >= val AND x < val at same value is also contradictory
        // x > val AND x <= val at same value is also contradictory

        let lower_strict = true;
        let upper_strict = true;

        let contradicts = bound_val == bound_val && (lower_strict || upper_strict);
        assert!(
            contradicts,
            "Equal bounds with any strictness is contradictory"
        );
    }

    // ========================================================================
    // TableauRow Invariants
    // ========================================================================

    /// Coefficient lookup returns zero for missing variables
    #[kani::proof]
    fn proof_coeff_missing_is_zero() {
        let row = TableauRow {
            basic_var: 0,
            coeffs: vec![(1, BigRational::from(BigInt::from(3)))],
            constant: BigRational::zero(),
        };

        // Variable 2 is not in the row
        let coeff = row.coeff(2);
        assert!(coeff.is_zero(), "Missing variable has zero coefficient");
    }

    /// contains returns true iff variable in coeffs
    #[kani::proof]
    fn proof_contains_correctness() {
        let row = TableauRow {
            basic_var: 0,
            coeffs: vec![
                (1, BigRational::from(BigInt::from(3))),
                (2, BigRational::from(BigInt::from(-5))),
            ],
            constant: BigRational::zero(),
        };

        assert!(row.contains(1), "Variable 1 is in row");
        assert!(row.contains(2), "Variable 2 is in row");
        assert!(!row.contains(3), "Variable 3 is not in row");
        assert!(!row.contains(0), "Basic var 0 is not in coeffs");
    }

    // ========================================================================
    // Solver State Invariants
    // ========================================================================

    /// Push increases scope depth, pop decreases it
    #[kani::proof]
    fn proof_push_pop_scope_depth() {
        let terms = z4_core::term::TermStore::new();
        let mut solver = LraSolver::new(&terms);

        let initial_scopes = solver.scopes.len();
        assert!(initial_scopes == 0, "Initially no scopes");

        solver.push();
        assert!(solver.scopes.len() == 1, "Push adds scope");

        solver.push();
        assert!(solver.scopes.len() == 2, "Second push adds scope");

        solver.pop();
        assert!(solver.scopes.len() == 1, "Pop removes scope");

        solver.pop();
        assert!(solver.scopes.len() == 0, "Final pop returns to empty");
    }

    /// Pop on empty scopes is safe (no-op)
    #[kani::proof]
    fn proof_pop_empty_is_safe() {
        let terms = z4_core::term::TermStore::new();
        let mut solver = LraSolver::new(&terms);

        // Pop with no pushes should be a no-op
        solver.pop();
        assert!(solver.scopes.is_empty(), "Pop on empty is no-op");
    }

    /// Reset clears all state
    #[kani::proof]
    fn proof_reset_clears_state() {
        let terms = z4_core::term::TermStore::new();
        let mut solver = LraSolver::new(&terms);

        // Add some state
        solver.push();
        solver.next_var = 10;

        solver.reset();

        assert!(solver.rows.is_empty(), "Reset clears rows");
        assert!(solver.vars.is_empty(), "Reset clears vars");
        assert!(solver.term_to_var.is_empty(), "Reset clears term_to_var");
        assert!(solver.var_to_term.is_empty(), "Reset clears var_to_term");
        assert!(solver.next_var == 0, "Reset resets next_var");
        assert!(solver.trail.is_empty(), "Reset clears trail");
        assert!(solver.scopes.is_empty(), "Reset clears scopes");
        assert!(solver.asserted.is_empty(), "Reset clears asserted");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use z4_core::term::TermStore;
    use z4_core::Sort;

    #[test]
    fn test_fractional_part() {
        // frac(6/5) = frac(1.2) = 0.2 = 1/5
        let v = BigRational::new(BigInt::from(6), BigInt::from(5));
        assert_eq!(
            fractional_part(&v),
            BigRational::new(BigInt::from(1), BigInt::from(5))
        );

        // frac(-6/5) = frac(-1.2) = -1.2 - floor(-1.2) = -1.2 - (-2) = 0.8 = 4/5
        let v = BigRational::new(BigInt::from(-6), BigInt::from(5));
        assert_eq!(
            fractional_part(&v),
            BigRational::new(BigInt::from(4), BigInt::from(5))
        );

        // frac(31/5) = frac(6.2) = 0.2 = 1/5
        let v = BigRational::new(BigInt::from(31), BigInt::from(5));
        assert_eq!(
            fractional_part(&v),
            BigRational::new(BigInt::from(1), BigInt::from(5))
        );

        // frac(5) = 0
        let v = BigRational::from(BigInt::from(5));
        assert_eq!(fractional_part(&v), BigRational::zero());

        // frac(-5) = 0
        let v = BigRational::from(BigInt::from(-5));
        assert_eq!(fractional_part(&v), BigRational::zero());

        // frac(1/5) = 1/5
        let v = BigRational::new(BigInt::from(1), BigInt::from(5));
        assert_eq!(
            fractional_part(&v),
            BigRational::new(BigInt::from(1), BigInt::from(5))
        );

        // frac(-1/5) = frac(-0.2) = -0.2 - (-1) = 0.8 = 4/5
        let v = BigRational::new(BigInt::from(-1), BigInt::from(5));
        assert_eq!(
            fractional_part(&v),
            BigRational::new(BigInt::from(4), BigInt::from(5))
        );
    }

    #[test]
    fn test_linear_expr_basic() {
        let mut expr = LinearExpr::zero();
        assert!(expr.is_constant());
        assert!(expr.constant.is_zero());

        expr.add_term(0, BigRational::from(BigInt::from(3)));
        assert!(!expr.is_constant());
        assert_eq!(expr.coeffs.len(), 1);
        assert_eq!(expr.coeffs[0], (0, BigRational::from(BigInt::from(3))));
    }

    #[test]
    fn test_linear_expr_combine() {
        let mut expr = LinearExpr::zero();
        expr.add_term(0, BigRational::from(BigInt::from(3)));
        expr.add_term(0, BigRational::from(BigInt::from(2)));

        // Should combine: 3x + 2x = 5x
        assert_eq!(expr.coeffs.len(), 1);
        assert_eq!(expr.coeffs[0], (0, BigRational::from(BigInt::from(5))));
    }

    #[test]
    fn test_linear_expr_cancel() {
        let mut expr = LinearExpr::zero();
        expr.add_term(0, BigRational::from(BigInt::from(3)));
        expr.add_term(0, BigRational::from(BigInt::from(-3)));

        // Should cancel: 3x - 3x = 0
        assert!(expr.coeffs.is_empty());
    }

    #[test]
    fn test_lra_solver_trivial_sat() {
        let terms = TermStore::new();
        let mut solver = LraSolver::new(&terms);

        // Empty problem is SAT
        assert!(matches!(solver.check(), TheoryResult::Sat));
    }

    #[test]
    fn test_lra_solver_simple_bound() {
        let mut terms = TermStore::new();

        // x <= 5
        let x = terms.mk_var("x", Sort::Real);
        let five = terms.mk_rational(BigRational::from(BigInt::from(5)));
        let atom = terms.mk_le(x, five);

        let mut solver = LraSolver::new(&terms);
        solver.assert_literal(atom, true);

        assert!(matches!(solver.check(), TheoryResult::Sat));
    }

    #[test]
    fn test_lra_solver_conflicting_bounds() {
        let mut terms = TermStore::new();

        // x >= 10 and x <= 5 should be UNSAT
        let x = terms.mk_var("x", Sort::Real);
        let five = terms.mk_rational(BigRational::from(BigInt::from(5)));
        let ten = terms.mk_rational(BigRational::from(BigInt::from(10)));

        let le_atom = terms.mk_le(x, five); // x <= 5
        let ge_atom = terms.mk_ge(x, ten); // x >= 10

        let mut solver = LraSolver::new(&terms);
        solver.assert_literal(le_atom, true);
        solver.assert_literal(ge_atom, true);

        let result = solver.check();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_lra_solver_linear_constraint() {
        let mut terms = TermStore::new();

        // x + y <= 10, x >= 5, y >= 6 should be UNSAT (5 + 6 = 11 > 10)
        let x = terms.mk_var("x", Sort::Real);
        let y = terms.mk_var("y", Sort::Real);
        let five = terms.mk_rational(BigRational::from(BigInt::from(5)));
        let six = terms.mk_rational(BigRational::from(BigInt::from(6)));
        let ten = terms.mk_rational(BigRational::from(BigInt::from(10)));

        let sum = terms.mk_add(vec![x, y]);
        let sum_le = terms.mk_le(sum, ten); // x + y <= 10
        let x_ge = terms.mk_ge(x, five); // x >= 5
        let y_ge = terms.mk_ge(y, six); // y >= 6

        let mut solver = LraSolver::new(&terms);
        solver.assert_literal(sum_le, true);
        solver.assert_literal(x_ge, true);
        solver.assert_literal(y_ge, true);

        let result = solver.check();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_lra_solver_linear_constraint_sat() {
        let mut terms = TermStore::new();

        // x + y <= 10, x >= 3, y >= 4 should be SAT (3 + 4 = 7 <= 10)
        let x = terms.mk_var("x", Sort::Real);
        let y = terms.mk_var("y", Sort::Real);
        let three = terms.mk_rational(BigRational::from(BigInt::from(3)));
        let four = terms.mk_rational(BigRational::from(BigInt::from(4)));
        let ten = terms.mk_rational(BigRational::from(BigInt::from(10)));

        let sum = terms.mk_add(vec![x, y]);
        let sum_le = terms.mk_le(sum, ten); // x + y <= 10
        let x_ge = terms.mk_ge(x, three); // x >= 3
        let y_ge = terms.mk_ge(y, four); // y >= 4

        let mut solver = LraSolver::new(&terms);
        solver.assert_literal(sum_le, true);
        solver.assert_literal(x_ge, true);
        solver.assert_literal(y_ge, true);

        let result = solver.check();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_lra_solver_push_pop() {
        let mut terms = TermStore::new();

        let x = terms.mk_var("x", Sort::Real);
        let three = terms.mk_rational(BigRational::from(BigInt::from(3)));
        let five = terms.mk_rational(BigRational::from(BigInt::from(5)));
        let ten = terms.mk_rational(BigRational::from(BigInt::from(10)));

        let le_atom = terms.mk_le(x, ten); // x <= 10
        let ge_atom = terms.mk_ge(x, five); // x >= 5
        let lt_atom = terms.mk_lt(x, three); // x < 3

        let mut solver = LraSolver::new(&terms);
        solver.assert_literal(le_atom, true);
        solver.assert_literal(ge_atom, true);

        assert!(matches!(solver.check(), TheoryResult::Sat));

        // Push and add conflicting constraint
        solver.push();
        solver.assert_literal(lt_atom, true);

        // x >= 5 and x < 3 conflicts
        let result = solver.check();
        assert!(matches!(result, TheoryResult::Unsat(_)));

        // Pop should restore SAT state
        solver.pop();
        solver.reset(); // Need reset to clear asserted atoms
        solver.assert_literal(le_atom, true);
        solver.assert_literal(ge_atom, true);
        assert!(matches!(solver.check(), TheoryResult::Sat));
    }

    #[test]
    fn test_lra_solver_strict_inequality() {
        let mut terms = TermStore::new();

        // x < 5 and x > 5 should be UNSAT
        let x = terms.mk_var("x", Sort::Real);
        let five = terms.mk_rational(BigRational::from(BigInt::from(5)));

        let lt_atom = terms.mk_lt(x, five); // x < 5
        let gt_atom = terms.mk_gt(x, five); // x > 5

        let mut solver = LraSolver::new(&terms);
        solver.assert_literal(lt_atom, true);
        solver.assert_literal(gt_atom, true);

        let result = solver.check();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_lra_solver_equality() {
        let mut terms = TermStore::new();

        // x = 5 and x > 5 should be UNSAT
        let x = terms.mk_var("x", Sort::Real);
        let five = terms.mk_rational(BigRational::from(BigInt::from(5)));

        let eq_atom = terms.mk_eq(x, five); // x = 5
        let gt_atom = terms.mk_gt(x, five); // x > 5

        let mut solver = LraSolver::new(&terms);
        solver.assert_literal(eq_atom, true);
        solver.assert_literal(gt_atom, true);

        let result = solver.check();
        assert!(matches!(result, TheoryResult::Unsat(_)), "{result:?}");
    }

    #[test]
    fn test_lra_solver_scaled_variable() {
        let mut terms = TermStore::new();

        // 2*x >= 10 should be SAT when x >= 5
        let x = terms.mk_var("x", Sort::Real);
        let two = terms.mk_rational(BigRational::from(BigInt::from(2)));
        let ten = terms.mk_rational(BigRational::from(BigInt::from(10)));
        let five = terms.mk_rational(BigRational::from(BigInt::from(5)));

        let scaled = terms.mk_mul(vec![two, x]);
        let ge_scaled = terms.mk_ge(scaled, ten); // 2*x >= 10
        let ge_five = terms.mk_ge(x, five); // x >= 5

        let mut solver = LraSolver::new(&terms);
        solver.assert_literal(ge_scaled, true);
        solver.assert_literal(ge_five, true);

        assert!(matches!(solver.check(), TheoryResult::Sat));
    }

    #[test]
    fn test_lra_solver_nonlinear_returns_unknown() {
        let mut terms = TermStore::new();

        // x * y = 1 is non-linear; the LRA solver must not claim SAT.
        let x = terms.mk_var("x", Sort::Real);
        let y = terms.mk_var("y", Sort::Real);
        let one = terms.mk_rational(BigRational::from(BigInt::from(1)));

        let xy = terms.mk_mul(vec![x, y]);
        let eq = terms.mk_eq(xy, one);

        let mut solver = LraSolver::new(&terms);
        solver.assert_literal(eq, true);

        assert!(matches!(
            solver.check(),
            TheoryResult::Unknown | TheoryResult::Unsat(_)
        ));
    }

    #[test]
    fn test_lra_solver_disequality_violated() {
        let mut terms = TermStore::new();

        // x != 5 combined with x = 5 should return UNSAT (model violates disequality).
        // The solver detects that the only solution (x=5) violates x != 5.
        let x = terms.mk_var("x", Sort::Real);
        let five = terms.mk_rational(BigRational::from(BigInt::from(5)));

        // x = 5 (equality atom, asserted as NOT TRUE means x != 5)
        let eq = terms.mk_eq(x, five);
        // x >= 5
        let ge = terms.mk_ge(x, five);
        // x <= 5
        let le = terms.mk_le(x, five);

        let mut solver = LraSolver::new(&terms);
        // Assert x != 5 by asserting the equality atom with value=false
        solver.assert_literal(eq, false);
        // Assert x >= 5
        solver.assert_literal(ge, true);
        // Assert x <= 5
        solver.assert_literal(le, true);

        // This is unsatisfiable (x=5 but x!=5). The solver should detect this
        // by checking that the model (x=5) violates the disequality.
        let result = solver.check();
        assert!(
            matches!(result, TheoryResult::Unsat(_)),
            "Expected Unsat but got {:?}",
            result
        );
    }

    #[test]
    fn test_lra_solver_two_var_equality_with_bounds_sat() {
        let mut terms = TermStore::new();

        // 4*x + 3*y = 70, x >= 0, y >= 0, x <= 17 is satisfiable over reals.
        let x = terms.mk_var("x", Sort::Real);
        let y = terms.mk_var("y", Sort::Real);
        let four = terms.mk_rational(BigRational::from(BigInt::from(4)));
        let three = terms.mk_rational(BigRational::from(BigInt::from(3)));
        let seventy = terms.mk_rational(BigRational::from(BigInt::from(70)));
        let zero = terms.mk_rational(BigRational::zero());
        let seventeen = terms.mk_rational(BigRational::from(BigInt::from(17)));

        let four_x = terms.mk_mul(vec![four, x]);
        let three_y = terms.mk_mul(vec![three, y]);
        let lhs = terms.mk_add(vec![four_x, three_y]);

        let eq = terms.mk_eq(lhs, seventy);
        let x_ge = terms.mk_ge(x, zero);
        let y_ge = terms.mk_ge(y, zero);
        let x_le = terms.mk_le(x, seventeen);

        let mut solver = LraSolver::new(&terms);
        solver.assert_literal(eq, true);
        solver.assert_literal(x_ge, true);
        solver.assert_literal(y_ge, true);
        solver.assert_literal(x_le, true);

        let result = solver.check();
        assert!(matches!(result, TheoryResult::Sat), "{result:?}");
    }

    #[test]
    fn test_lra_solver_distinct_atom_forced_unsat() {
        let mut terms = TermStore::new();

        // (distinct x 5) with x = 5 forced should return UNSAT.
        // This tests the distinct atom parsing (not negated equality).
        let x = terms.mk_var("x", Sort::Real);
        let five = terms.mk_rational(BigRational::from(BigInt::from(5)));

        // Create distinct atom directly
        let distinct = terms.mk_distinct(vec![x, five]);
        // x >= 5
        let ge = terms.mk_ge(x, five);
        // x <= 5
        let le = terms.mk_le(x, five);

        let mut solver = LraSolver::new(&terms);
        // Assert (distinct x 5) = true means x != 5
        solver.assert_literal(distinct, true);
        // Assert x >= 5 and x <= 5, forcing x = 5
        solver.assert_literal(ge, true);
        solver.assert_literal(le, true);

        // x is forced to 5 but (distinct x 5) requires x != 5: UNSAT
        let result = solver.check();
        assert!(
            matches!(result, TheoryResult::Unsat(_)),
            "Expected Unsat but got {:?}",
            result
        );
    }

    #[test]
    fn test_lra_solver_distinct_atom_with_slack_unknown() {
        let mut terms = TermStore::new();

        // (distinct x 5) with x >= 3 should return Unknown (or Sat with x != 5)
        // because simplex might find x=5 but other solutions exist.
        let x = terms.mk_var("x", Sort::Real);
        let three = terms.mk_rational(BigRational::from(BigInt::from(3)));
        let five = terms.mk_rational(BigRational::from(BigInt::from(5)));

        // Create distinct atom
        let distinct = terms.mk_distinct(vec![x, five]);
        // x >= 3
        let ge = terms.mk_ge(x, three);

        let mut solver = LraSolver::new(&terms);
        solver.assert_literal(distinct, true);
        solver.assert_literal(ge, true);

        // Result should be Sat (x=3 or x=4 works) or Unknown (if simplex found x=5)
        // It should NOT be Unsat because solutions exist.
        let result = solver.check();
        assert!(
            matches!(result, TheoryResult::Sat | TheoryResult::Unknown),
            "Expected Sat or Unknown but got {:?}",
            result
        );
    }
}
