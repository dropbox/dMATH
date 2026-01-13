//! Linear Rational Arithmetic (LRA) theory solver using Simplex
//!
//! This module implements a theory solver for linear arithmetic over rationals
//! using the dual simplex algorithm (as used in Z3 and CVC5).
//!
//! # Theory of Linear Rational Arithmetic (LRA)
//!
//! LRA handles:
//! - Linear constraints: `a₁x₁ + a₂x₂ + ... + aₙxₙ ≤ c`
//! - Strict inequalities: `a₁x₁ + a₂x₂ + ... + aₙxₙ < c`
//! - Equalities: `a₁x₁ + a₂x₂ + ... + aₙxₙ = c`
//!
//! # Simplex Algorithm
//!
//! The solver maintains a tableau in the form:
//! ```text
//! s₁ = a₁₁x₁ + a₁₂x₂ + ... (slack variable for row 1)
//! s₂ = a₂₁x₁ + a₂₂x₂ + ... (slack variable for row 2)
//! ```
//!
//! Basic variables (left side) have their values determined by non-basic variables.
//! The algorithm pivots variables between basic and non-basic to satisfy bounds.
//!
//! # Implementation Notes
//!
//! - Uses rational arithmetic for exactness (no floating point errors)
//! - Supports incremental solving with backtracking
//! - Detects conflicts and generates explanations

use crate::cdcl::Lit;
use crate::smt::{SmtTerm, TermId, TheoryCheckResult, TheoryLiteral, TheorySolver};
use std::collections::HashMap;

/// A rational number represented as numerator/denominator
/// Using i64 for simplicity; a production implementation would use arbitrary precision
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rational {
    num: i64,
    den: i64, // Always positive
}

impl Rational {
    pub const ZERO: Rational = Rational { num: 0, den: 1 };
    pub const ONE: Rational = Rational { num: 1, den: 1 };

    /// Create a new rational number, automatically normalized
    pub fn new(num: i64, den: i64) -> Self {
        assert!(den != 0, "Division by zero in Rational::new");
        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
        let g = Self::gcd(num.unsigned_abs(), den.unsigned_abs());
        Rational {
            num: num / (g as i64),
            den: den / (g as i64),
        }
    }

    /// Create from an integer
    pub fn from_int(n: i64) -> Self {
        Rational { num: n, den: 1 }
    }

    /// GCD using Euclidean algorithm
    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        if a == 0 {
            1
        } else {
            a
        }
    }

    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    pub fn is_positive(&self) -> bool {
        self.num > 0
    }

    pub fn is_negative(&self) -> bool {
        self.num < 0
    }

    #[must_use]
    pub fn abs(&self) -> Self {
        Rational {
            num: self.num.abs(),
            den: self.den,
        }
    }

    #[must_use]
    pub fn neg(&self) -> Self {
        Rational {
            num: -self.num,
            den: self.den,
        }
    }

    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Rational::new(
            self.num * other.den + other.num * self.den,
            self.den * other.den,
        )
    }

    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        Rational::new(
            self.num * other.den - other.num * self.den,
            self.den * other.den,
        )
    }

    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        Rational::new(self.num * other.num, self.den * other.den)
    }

    #[must_use]
    pub fn div(&self, other: &Self) -> Self {
        assert!(other.num != 0, "Division by zero in Rational::div");
        Rational::new(self.num * other.den, self.den * other.num)
    }
}

impl std::cmp::PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for Rational {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // a/b vs c/d  =>  a*d vs c*b
        let lhs = self.num * other.den;
        let rhs = other.num * self.den;
        lhs.cmp(&rhs)
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// Variable identifier in the arithmetic theory
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArithVar(pub u32);

/// Type of bound
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundType {
    /// x ≤ c
    Upper,
    /// x ≥ c
    Lower,
}

/// A bound on a variable with its justification literal
#[derive(Clone, Debug)]
pub struct Bound {
    pub value: Rational,
    pub bound_type: BoundType,
    /// Whether this is a strict inequality (< or >, not ≤ or ≥)
    pub strict: bool,
    /// The literal that justified this bound
    pub reason: Lit,
    /// Decision level at which this bound was added
    pub level: u32,
}

/// Linear expression: Σ aᵢxᵢ
#[derive(Clone, Debug, Default)]
pub struct LinearExpr {
    /// Coefficients indexed by variable
    pub coeffs: HashMap<ArithVar, Rational>,
}

impl LinearExpr {
    pub fn new() -> Self {
        LinearExpr {
            coeffs: HashMap::new(),
        }
    }

    /// Create expression with single variable coefficient
    pub fn var(v: ArithVar) -> Self {
        let mut expr = LinearExpr::new();
        expr.coeffs.insert(v, Rational::ONE);
        expr
    }

    /// Add a term: expr += coeff * var
    pub fn add_term(&mut self, var: ArithVar, coeff: Rational) {
        let entry = self.coeffs.entry(var).or_insert(Rational::ZERO);
        *entry = entry.add(&coeff);
        if entry.is_zero() {
            self.coeffs.remove(&var);
        }
    }

    /// Multiply the entire expression by a scalar
    pub fn scale(&mut self, scalar: &Rational) {
        for coeff in self.coeffs.values_mut() {
            *coeff = coeff.mul(scalar);
        }
    }

    /// Add another expression: self += other
    pub fn add_expr(&mut self, other: &LinearExpr) {
        for (&var, coeff) in &other.coeffs {
            self.add_term(var, *coeff);
        }
    }

    /// Evaluate expression given variable assignments
    pub fn evaluate(&self, assignment: &HashMap<ArithVar, Rational>) -> Rational {
        let mut sum = Rational::ZERO;
        for (&var, coeff) in &self.coeffs {
            if let Some(val) = assignment.get(&var) {
                sum = sum.add(&coeff.mul(val));
            }
        }
        sum
    }
}

/// Row in the simplex tableau
/// Represents: basic_var = constant + Σ coeffᵢ * non_basicᵢ
#[derive(Clone, Debug)]
struct TableauRow {
    /// The basic variable for this row
    basic_var: ArithVar,
    /// Constant term
    constant: Rational,
    /// Coefficients for non-basic variables
    coeffs: HashMap<ArithVar, Rational>,
}

impl TableauRow {
    /// Evaluate the row given non-basic variable values
    fn evaluate(&self, assignment: &HashMap<ArithVar, Rational>) -> Rational {
        let mut sum = self.constant;
        for (&var, coeff) in &self.coeffs {
            if let Some(val) = assignment.get(&var) {
                sum = sum.add(&coeff.mul(val));
            }
        }
        sum
    }
}

/// Linear Rational Arithmetic theory solver
pub struct ArithmeticTheory {
    /// Number of original variables
    num_vars: u32,
    /// Number of slack variables
    num_slack: u32,
    /// Tableau rows (one per basic variable)
    tableau: Vec<TableauRow>,
    /// Current variable assignment (for non-basic vars)
    assignment: HashMap<ArithVar, Rational>,
    /// Lower bounds on variables
    lower_bounds: HashMap<ArithVar, Bound>,
    /// Upper bounds on variables
    upper_bounds: HashMap<ArithVar, Bound>,
    /// Trail of bounds for backtracking: (level, var, was_lower, old_bound)
    bound_trail: Vec<(u32, ArithVar, bool, Option<Bound>)>,
    /// Current decision level
    level: u32,
    /// Mapping from SMT term IDs to arithmetic variables
    term_to_var: HashMap<TermId, ArithVar>,
    /// SMT terms (shared reference)
    terms: Vec<SmtTerm>,
}

impl ArithmeticTheory {
    /// Create a new arithmetic theory solver
    pub fn new() -> Self {
        ArithmeticTheory {
            num_vars: 0,
            num_slack: 0,
            tableau: Vec::new(),
            assignment: HashMap::new(),
            lower_bounds: HashMap::new(),
            upper_bounds: HashMap::new(),
            bound_trail: Vec::new(),
            level: 0,
            term_to_var: HashMap::new(),
            terms: Vec::new(),
        }
    }

    /// Set the terms (called by SMT solver)
    pub fn set_terms(&mut self, terms: Vec<SmtTerm>) {
        self.terms = terms;
    }

    /// Create or get a variable for an SMT term
    fn get_or_create_var(&mut self, term_id: TermId) -> ArithVar {
        if let Some(&var) = self.term_to_var.get(&term_id) {
            return var;
        }

        let var = ArithVar(self.num_vars);
        self.num_vars += 1;
        self.term_to_var.insert(term_id, var);

        // Initialize assignment to 0
        self.assignment.insert(var, Rational::ZERO);

        var
    }

    /// Create a new slack variable
    fn new_slack_var(&mut self) -> ArithVar {
        let var = ArithVar(self.num_vars + self.num_slack);
        self.num_slack += 1;
        self.assignment.insert(var, Rational::ZERO);
        var
    }

    /// Add a constraint: expr ≤ c (or < c if strict)
    /// Returns a slack variable representing the constraint
    fn add_upper_constraint(
        &mut self,
        expr: LinearExpr,
        bound: Rational,
        strict: bool,
        reason: Lit,
    ) -> TheoryCheckResult {
        // Create slack variable: s = expr
        // Constraint becomes: s ≤ bound (or s < bound)
        let slack = self.new_slack_var();

        // Add tableau row: slack = Σ coeffᵢ * xᵢ
        let row = TableauRow {
            basic_var: slack,
            constant: Rational::ZERO,
            coeffs: expr.coeffs.clone(),
        };
        self.tableau.push(row);

        // Add upper bound on slack
        self.assert_upper_bound(slack, bound, strict, reason)
    }

    /// Add a constraint: expr ≥ c (or > c if strict)
    #[allow(dead_code)]
    fn add_lower_constraint(
        &mut self,
        expr: LinearExpr,
        bound: Rational,
        strict: bool,
        reason: Lit,
    ) -> TheoryCheckResult {
        let slack = self.new_slack_var();

        let row = TableauRow {
            basic_var: slack,
            constant: Rational::ZERO,
            coeffs: expr.coeffs.clone(),
        };
        self.tableau.push(row);

        self.assert_lower_bound(slack, bound, strict, reason)
    }

    /// Assert an upper bound: var ≤ bound
    fn assert_upper_bound(
        &mut self,
        var: ArithVar,
        bound: Rational,
        strict: bool,
        reason: Lit,
    ) -> TheoryCheckResult {
        // Check for immediate conflict with lower bound
        if let Some(lower) = self.lower_bounds.get(&var) {
            let conflict = if strict || lower.strict {
                bound <= lower.value
            } else {
                bound < lower.value
            };
            if conflict {
                // Conflict: lower > upper
                return TheoryCheckResult::Conflict(vec![lower.reason, reason]);
            }
        }

        // Save old bound for backtracking
        let old_bound = self.upper_bounds.get(&var).cloned();
        self.bound_trail
            .push((self.level, var, false, old_bound.clone()));

        // Update bound if tighter
        // For upper bounds: smaller is tighter
        // strict < is tighter than <= at the same value
        let should_update = match &old_bound {
            None => true,
            Some(old) => {
                if bound < old.value {
                    true
                } else if bound == old.value {
                    // At same value, strict is tighter
                    strict && !old.strict
                } else {
                    false
                }
            }
        };

        if should_update {
            self.upper_bounds.insert(
                var,
                Bound {
                    value: bound,
                    bound_type: BoundType::Upper,
                    strict,
                    reason,
                    level: self.level,
                },
            );
        }

        // Check if we need to repair assignment
        self.check_and_repair()
    }

    /// Assert a lower bound: var ≥ bound
    #[allow(dead_code)]
    fn assert_lower_bound(
        &mut self,
        var: ArithVar,
        bound: Rational,
        strict: bool,
        reason: Lit,
    ) -> TheoryCheckResult {
        // Check for immediate conflict with upper bound
        if let Some(upper) = self.upper_bounds.get(&var) {
            let conflict = if strict || upper.strict {
                bound >= upper.value
            } else {
                bound > upper.value
            };
            if conflict {
                return TheoryCheckResult::Conflict(vec![reason, upper.reason]);
            }
        }

        // Save old bound for backtracking
        let old_bound = self.lower_bounds.get(&var).cloned();
        self.bound_trail
            .push((self.level, var, true, old_bound.clone()));

        // Update bound if tighter
        // For lower bounds: larger is tighter
        // strict > is tighter than >= at the same value
        let should_update = match &old_bound {
            None => true,
            Some(old) => {
                if bound > old.value {
                    true
                } else if bound == old.value {
                    // At same value, strict is tighter
                    strict && !old.strict
                } else {
                    false
                }
            }
        };

        if should_update {
            self.lower_bounds.insert(
                var,
                Bound {
                    value: bound,
                    bound_type: BoundType::Lower,
                    strict,
                    reason,
                    level: self.level,
                },
            );
        }

        self.check_and_repair()
    }

    /// Check bounds and repair assignment using simplex
    fn check_and_repair(&mut self) -> TheoryCheckResult {
        // First, fix any non-basic variable bound violations by updating assignment directly
        // Non-basic variables are not in the tableau, so we can adjust them freely
        // (as long as they don't affect basic variables violating their bounds)
        self.fix_nonbasic_bounds();

        // Dual simplex: find a basic variable that violates its bounds
        loop {
            let violation = self.find_violated_basic();
            match violation {
                None => return TheoryCheckResult::Consistent,
                Some((basic_var, is_lower_violation)) => {
                    // Try to pivot to fix the violation
                    if let Some(non_basic) = self.find_pivot(basic_var, is_lower_violation) {
                        self.pivot(basic_var, non_basic);
                    } else {
                        // No pivot possible - conflict!
                        let conflict = self.explain_conflict(basic_var, is_lower_violation);
                        return TheoryCheckResult::Conflict(conflict);
                    }
                }
            }
        }
    }

    /// Fix non-basic variable assignments to satisfy their bounds
    fn fix_nonbasic_bounds(&mut self) {
        // Collect non-basic variables (those not in any tableau row as basic)
        let basic_vars: std::collections::HashSet<ArithVar> =
            self.tableau.iter().map(|r| r.basic_var).collect();

        // Check each variable in assignment
        let vars: Vec<ArithVar> = self.assignment.keys().copied().collect();
        for var in vars {
            if basic_vars.contains(&var) {
                continue; // Skip basic variables, they're handled by simplex
            }

            let current = self.assignment.get(&var).copied().unwrap_or(Rational::ZERO);

            // Check lower bound
            if let Some(lower) = self.lower_bounds.get(&var) {
                let violated = if lower.strict {
                    current <= lower.value
                } else {
                    current < lower.value
                };
                if violated {
                    // Update assignment to satisfy lower bound
                    // For strict bound, we'd need infinitesimals; use bound value for now
                    self.assignment.insert(var, lower.value);
                }
            }

            // Check upper bound (after possibly updating from lower)
            let current = self.assignment.get(&var).copied().unwrap_or(Rational::ZERO);
            if let Some(upper) = self.upper_bounds.get(&var) {
                let violated = if upper.strict {
                    current >= upper.value
                } else {
                    current > upper.value
                };
                if violated {
                    self.assignment.insert(var, upper.value);
                }
            }
        }
    }

    /// Find a basic variable that violates its bounds
    fn find_violated_basic(&self) -> Option<(ArithVar, bool)> {
        for row in &self.tableau {
            let basic_var = row.basic_var;
            let value = row.evaluate(&self.assignment);

            // Check lower bound
            if let Some(lower) = self.lower_bounds.get(&basic_var) {
                let violated = if lower.strict {
                    value <= lower.value
                } else {
                    value < lower.value
                };
                if violated {
                    return Some((basic_var, true)); // lower violation
                }
            }

            // Check upper bound
            if let Some(upper) = self.upper_bounds.get(&basic_var) {
                let violated = if upper.strict {
                    value >= upper.value
                } else {
                    value > upper.value
                };
                if violated {
                    return Some((basic_var, false)); // upper violation
                }
            }
        }
        None
    }

    /// Find a non-basic variable to pivot with
    fn find_pivot(&self, basic_var: ArithVar, is_lower_violation: bool) -> Option<ArithVar> {
        // Find the row for this basic variable
        let row = self.tableau.iter().find(|r| r.basic_var == basic_var)?;

        // If lower bound violation: basic_var is too low
        // We need to increase it by pivoting with a non-basic that can increase it
        // If upper bound violation: basic_var is too high
        // We need to decrease it by pivoting with a non-basic that can decrease it

        for (&non_basic, coeff) in &row.coeffs {
            if coeff.is_zero() {
                continue;
            }

            // Check if changing non_basic can help
            let can_increase = self.can_increase(non_basic);
            let can_decrease = self.can_decrease(non_basic);

            if is_lower_violation {
                // Need to increase basic_var
                // If coeff > 0: increase non_basic to increase basic
                // If coeff < 0: decrease non_basic to increase basic
                if (coeff.is_positive() && can_increase) || (coeff.is_negative() && can_decrease) {
                    return Some(non_basic);
                }
            } else {
                // Need to decrease basic_var
                // If coeff > 0: decrease non_basic to decrease basic
                // If coeff < 0: increase non_basic to decrease basic
                if (coeff.is_positive() && can_decrease) || (coeff.is_negative() && can_increase) {
                    return Some(non_basic);
                }
            }
        }

        None
    }

    /// Check if a non-basic variable can increase
    fn can_increase(&self, var: ArithVar) -> bool {
        let current = self.assignment.get(&var).copied().unwrap_or(Rational::ZERO);
        match self.upper_bounds.get(&var) {
            None => true,
            Some(bound) => {
                if bound.strict {
                    current < bound.value
                } else {
                    current <= bound.value
                }
            }
        }
    }

    /// Check if a non-basic variable can decrease
    fn can_decrease(&self, var: ArithVar) -> bool {
        let current = self.assignment.get(&var).copied().unwrap_or(Rational::ZERO);
        match self.lower_bounds.get(&var) {
            None => true,
            Some(bound) => {
                if bound.strict {
                    current > bound.value
                } else {
                    current >= bound.value
                }
            }
        }
    }

    /// Perform a pivot operation: swap basic and non-basic variables
    fn pivot(&mut self, basic_var: ArithVar, non_basic: ArithVar) {
        // Find the row index for basic_var
        let Some(row_idx) = self.tableau.iter().position(|r| r.basic_var == basic_var) else {
            return;
        };

        // Get coefficient of non_basic in the row
        let coeff = match self.tableau[row_idx].coeffs.get(&non_basic).copied() {
            Some(c) if !c.is_zero() => c,
            _ => return,
        };

        // Rewrite row to express non_basic in terms of basic_var and other non-basics
        // Originally: basic_var = c + Σ aᵢ xᵢ + coeff * non_basic
        // New: non_basic = (basic_var - c - Σ aᵢ xᵢ) / coeff
        //                = basic_var/coeff - c/coeff - Σ (aᵢ/coeff) xᵢ

        let inv_coeff = Rational::ONE.div(&coeff);
        let old_constant = self.tableau[row_idx].constant;

        // Build new row for non_basic
        let mut new_coeffs = HashMap::new();

        // basic_var gets coefficient 1/coeff
        new_coeffs.insert(basic_var, inv_coeff);

        // Other non-basics get -aᵢ/coeff
        for (&var, &c) in &self.tableau[row_idx].coeffs {
            if var != non_basic {
                new_coeffs.insert(var, c.neg().mul(&inv_coeff));
            }
        }

        let new_row = TableauRow {
            basic_var: non_basic,
            constant: old_constant.neg().mul(&inv_coeff),
            coeffs: new_coeffs,
        };

        // Update the row
        self.tableau[row_idx] = new_row.clone();

        // Substitute into other rows
        for i in 0..self.tableau.len() {
            if i == row_idx {
                continue;
            }

            if let Some(&c) = self.tableau[i].coeffs.get(&non_basic) {
                if c.is_zero() {
                    continue;
                }

                // This row has non_basic with coefficient c
                // Substitute: non_basic = new_row
                // Result: add c * new_row to this row

                self.tableau[i].coeffs.remove(&non_basic);
                self.tableau[i].constant = self.tableau[i].constant.add(&c.mul(&new_row.constant));

                for (&var, &coef) in &new_row.coeffs {
                    let existing = self.tableau[i]
                        .coeffs
                        .get(&var)
                        .copied()
                        .unwrap_or(Rational::ZERO);
                    let new_val = existing.add(&c.mul(&coef));
                    if new_val.is_zero() {
                        self.tableau[i].coeffs.remove(&var);
                    } else {
                        self.tableau[i].coeffs.insert(var, new_val);
                    }
                }
            }
        }

        // Update assignment: set non_basic to its new value (to satisfy basic_var's bound)
        // The new value should be computed from the constraint being fixed
        let basic_val = self
            .assignment
            .get(&basic_var)
            .copied()
            .unwrap_or(Rational::ZERO);

        // Determine target value for basic variable
        let target = if let Some(lower) = self.lower_bounds.get(&basic_var) {
            if basic_val < lower.value {
                lower.value
            } else if let Some(upper) = self.upper_bounds.get(&basic_var) {
                if basic_val > upper.value {
                    upper.value
                } else {
                    basic_val
                }
            } else {
                basic_val
            }
        } else if let Some(upper) = self.upper_bounds.get(&basic_var) {
            if basic_val > upper.value {
                upper.value
            } else {
                basic_val
            }
        } else {
            basic_val
        };

        // Compute the change needed
        let delta = target.sub(&basic_val);
        let non_basic_delta = delta.div(&coeff);

        let non_basic_val = self
            .assignment
            .get(&non_basic)
            .copied()
            .unwrap_or(Rational::ZERO);
        self.assignment
            .insert(non_basic, non_basic_val.add(&non_basic_delta));
    }

    /// Explain a conflict: return the literals that led to it
    fn explain_conflict(&self, _basic_var: ArithVar, _is_lower_violation: bool) -> Vec<Lit> {
        // Collect all bounds that contributed to the conflict
        // For a full explanation, we'd trace back through pivots
        // For now, return all current bounds as a simple explanation
        let mut conflict = Vec::new();

        for bound in self.lower_bounds.values() {
            if bound.level <= self.level {
                conflict.push(bound.reason);
            }
        }

        for bound in self.upper_bounds.values() {
            if bound.level <= self.level {
                conflict.push(bound.reason);
            }
        }

        conflict
    }

    /// Process theory literal for less-than or less-equal
    fn handle_comparison(
        &mut self,
        t1: TermId,
        t2: TermId,
        is_le: bool,
        lit: Lit,
    ) -> TheoryCheckResult {
        // t1 < t2 or t1 ≤ t2
        // Equivalent to: t1 - t2 < 0 or t1 - t2 ≤ 0

        let v1 = self.get_or_create_var(t1);
        let v2 = self.get_or_create_var(t2);

        let mut expr = LinearExpr::new();
        expr.add_term(v1, Rational::ONE);
        expr.add_term(v2, Rational::ONE.neg());

        // t1 - t2 ≤ 0 (non-strict) or t1 - t2 < 0 (strict)
        self.add_upper_constraint(expr, Rational::ZERO, !is_le, lit)
    }

    /// Check if current assignment satisfies all bounds
    pub fn is_consistent(&self) -> bool {
        // Check all basic variables
        for row in &self.tableau {
            let value = row.evaluate(&self.assignment);

            if let Some(lower) = self.lower_bounds.get(&row.basic_var) {
                let violated = if lower.strict {
                    value <= lower.value
                } else {
                    value < lower.value
                };
                if violated {
                    return false;
                }
            }

            if let Some(upper) = self.upper_bounds.get(&row.basic_var) {
                let violated = if upper.strict {
                    value >= upper.value
                } else {
                    value > upper.value
                };
                if violated {
                    return false;
                }
            }
        }

        // Check all non-basic variables
        for (&var, &value) in &self.assignment {
            if let Some(lower) = self.lower_bounds.get(&var) {
                let violated = if lower.strict {
                    value <= lower.value
                } else {
                    value < lower.value
                };
                if violated {
                    return false;
                }
            }

            if let Some(upper) = self.upper_bounds.get(&var) {
                let violated = if upper.strict {
                    value >= upper.value
                } else {
                    value > upper.value
                };
                if violated {
                    return false;
                }
            }
        }

        true
    }

    /// Get statistics
    pub fn stats(&self) -> ArithStats {
        ArithStats {
            num_vars: self.num_vars as usize,
            num_slack: self.num_slack as usize,
            num_rows: self.tableau.len(),
            num_lower_bounds: self.lower_bounds.len(),
            num_upper_bounds: self.upper_bounds.len(),
        }
    }
}

impl Default for ArithmeticTheory {
    fn default() -> Self {
        Self::new()
    }
}

impl TheorySolver for ArithmeticTheory {
    fn assert_literal(&mut self, lit: Lit, theory_lit: &TheoryLiteral) -> TheoryCheckResult {
        match theory_lit {
            TheoryLiteral::Lt(t1, t2) => {
                // t1 < t2 (strict)
                self.handle_comparison(*t1, *t2, false, lit)
            }
            TheoryLiteral::Le(t1, t2) => {
                // t1 ≤ t2 (non-strict)
                self.handle_comparison(*t1, *t2, true, lit)
            }
            // Equality and disequality are handled by EUF theory
            // We could also handle arithmetic equality here as two bounds
            _ => TheoryCheckResult::Consistent,
        }
    }

    fn check(&self) -> TheoryCheckResult {
        if self.is_consistent() {
            TheoryCheckResult::Consistent
        } else {
            // Should not happen if incremental checking works correctly
            TheoryCheckResult::Conflict(vec![])
        }
    }

    fn backtrack(&mut self, level: u32) {
        if level >= self.level {
            return;
        }

        // Undo bounds added after target level
        while let Some(&(bound_level, var, is_lower, ref old_bound)) = self.bound_trail.last() {
            if bound_level <= level {
                break;
            }

            // Restore old bound
            if is_lower {
                match old_bound {
                    Some(b) => {
                        self.lower_bounds.insert(var, b.clone());
                    }
                    None => {
                        self.lower_bounds.remove(&var);
                    }
                }
            } else {
                match old_bound {
                    Some(b) => {
                        self.upper_bounds.insert(var, b.clone());
                    }
                    None => {
                        self.upper_bounds.remove(&var);
                    }
                }
            }

            self.bound_trail.pop();
        }

        self.level = level;
    }

    fn push(&mut self) {
        self.level += 1;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "LRA"
    }
}

/// Statistics for arithmetic theory
#[derive(Clone, Debug, Default)]
pub struct ArithStats {
    pub num_vars: usize,
    pub num_slack: usize,
    pub num_rows: usize,
    pub num_lower_bounds: usize,
    pub num_upper_bounds: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdcl::Var;

    fn make_lit(idx: u32, pos: bool) -> Lit {
        let var = Var::new(idx);
        if pos {
            Lit::pos(var)
        } else {
            Lit::neg(var)
        }
    }

    #[test]
    fn test_rational_basic() {
        let a = Rational::new(1, 2);
        let b = Rational::new(1, 3);

        // 1/2 + 1/3 = 5/6
        let sum = a.add(&b);
        assert_eq!(sum, Rational::new(5, 6));

        // 1/2 - 1/3 = 1/6
        let diff = a.sub(&b);
        assert_eq!(diff, Rational::new(1, 6));

        // 1/2 * 1/3 = 1/6
        let prod = a.mul(&b);
        assert_eq!(prod, Rational::new(1, 6));

        // 1/2 / 1/3 = 3/2
        let quot = a.div(&b);
        assert_eq!(quot, Rational::new(3, 2));
    }

    #[test]
    fn test_rational_comparison() {
        let a = Rational::new(1, 2);
        let b = Rational::new(1, 3);
        let c = Rational::new(2, 4); // = 1/2

        assert!(a > b);
        assert!(b < a);
        assert_eq!(a, c);
    }

    #[test]
    fn test_rational_normalization() {
        let a = Rational::new(2, 4);
        let b = Rational::new(-3, -6);
        let c = Rational::new(-2, 4);

        assert_eq!(a, Rational::new(1, 2));
        assert_eq!(b, Rational::new(1, 2));
        assert_eq!(c, Rational::new(-1, 2));
    }

    #[test]
    fn test_arithmetic_basic_upper() {
        let mut arith = ArithmeticTheory::new();

        // Create variable x
        let x = TermId(0);
        let v = arith.get_or_create_var(x);

        // Assert x ≤ 5
        let result = arith.assert_upper_bound(v, Rational::from_int(5), false, make_lit(0, true));
        assert!(matches!(result, TheoryCheckResult::Consistent));

        assert!(arith.is_consistent());
    }

    #[test]
    fn test_arithmetic_basic_lower() {
        let mut arith = ArithmeticTheory::new();

        let x = TermId(0);
        let v = arith.get_or_create_var(x);

        // Assert x ≥ 3
        let result = arith.assert_lower_bound(v, Rational::from_int(3), false, make_lit(0, true));
        assert!(matches!(result, TheoryCheckResult::Consistent));

        assert!(arith.is_consistent());
    }

    #[test]
    fn test_arithmetic_conflict() {
        let mut arith = ArithmeticTheory::new();

        let x = TermId(0);
        let v = arith.get_or_create_var(x);

        // Assert x ≤ 3
        let result = arith.assert_upper_bound(v, Rational::from_int(3), false, make_lit(0, true));
        assert!(matches!(result, TheoryCheckResult::Consistent));

        // Assert x ≥ 5 - should conflict
        let result = arith.assert_lower_bound(v, Rational::from_int(5), false, make_lit(1, true));
        assert!(matches!(result, TheoryCheckResult::Conflict(_)));
    }

    #[test]
    fn test_arithmetic_strict_conflict() {
        let mut arith = ArithmeticTheory::new();

        let x = TermId(0);
        let v = arith.get_or_create_var(x);

        // Assert x ≤ 3
        let result = arith.assert_upper_bound(v, Rational::from_int(3), false, make_lit(0, true));
        assert!(matches!(result, TheoryCheckResult::Consistent));

        // Assert x > 3 (strictly greater) - should conflict
        let result = arith.assert_lower_bound(v, Rational::from_int(3), true, make_lit(1, true));
        assert!(matches!(result, TheoryCheckResult::Conflict(_)));
    }

    #[test]
    fn test_arithmetic_non_strict_ok() {
        let mut arith = ArithmeticTheory::new();

        let x = TermId(0);
        let v = arith.get_or_create_var(x);

        // Assert x ≤ 3
        arith.assert_upper_bound(v, Rational::from_int(3), false, make_lit(0, true));

        // Assert x ≥ 3 (not strictly) - should be OK (x = 3)
        let result = arith.assert_lower_bound(v, Rational::from_int(3), false, make_lit(1, true));
        assert!(matches!(result, TheoryCheckResult::Consistent));
    }

    #[test]
    fn test_arithmetic_constraint() {
        let mut arith = ArithmeticTheory::new();

        // x < y using theory literals
        let x = TermId(0);
        let y = TermId(1);

        let result = arith.assert_literal(make_lit(0, true), &TheoryLiteral::Lt(x, y));
        assert!(matches!(result, TheoryCheckResult::Consistent));
    }

    #[test]
    fn test_arithmetic_backtrack() {
        let mut arith = ArithmeticTheory::new();

        let x = TermId(0);
        let v = arith.get_or_create_var(x);

        // Level 0: x ≤ 10
        arith.assert_upper_bound(v, Rational::from_int(10), false, make_lit(0, true));

        // Push to level 1
        arith.push();

        // Level 1: x ≤ 5 (tighter)
        arith.assert_upper_bound(v, Rational::from_int(5), false, make_lit(1, true));

        // x should have upper bound 5
        assert!(arith.upper_bounds.get(&v).map(|b| b.value) == Some(Rational::from_int(5)));

        // Backtrack to level 0
        arith.backtrack(0);

        // x should have upper bound 10 again
        assert!(arith.upper_bounds.get(&v).map(|b| b.value) == Some(Rational::from_int(10)));
    }

    #[test]
    fn test_linear_expr() {
        let mut expr = LinearExpr::new();
        let x = ArithVar(0);
        let y = ArithVar(1);

        // 2x + 3y
        expr.add_term(x, Rational::from_int(2));
        expr.add_term(y, Rational::from_int(3));

        let mut assignment = HashMap::new();
        assignment.insert(x, Rational::from_int(1));
        assignment.insert(y, Rational::from_int(2));

        // 2*1 + 3*2 = 8
        let result = expr.evaluate(&assignment);
        assert_eq!(result, Rational::from_int(8));
    }

    #[test]
    fn test_linear_expr_scale() {
        let mut expr = LinearExpr::new();
        let x = ArithVar(0);

        // 2x
        expr.add_term(x, Rational::from_int(2));

        // Scale by 3: 6x
        expr.scale(&Rational::from_int(3));

        assert_eq!(expr.coeffs.get(&x), Some(&Rational::from_int(6)));
    }

    #[test]
    fn test_stats() {
        let mut arith = ArithmeticTheory::new();

        let x = TermId(0);
        let y = TermId(1);
        let _ = arith.get_or_create_var(x);
        let _ = arith.get_or_create_var(y);

        let stats = arith.stats();
        assert_eq!(stats.num_vars, 2);
    }
}
