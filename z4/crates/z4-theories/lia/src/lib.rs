//! Z4 LIA - Linear Integer Arithmetic theory solver
//!
//! Implements branch-and-bound over LRA for integer arithmetic,
//! following the DPLL(T) approach where the SAT solver handles branching.
//!
//! ## Algorithm Overview
//!
//! The solver uses lazy branch-and-bound with cutting planes:
//!
//! 1. Solve the LRA (Linear Real Arithmetic) relaxation
//! 2. If UNSAT, return UNSAT (integers can't satisfy it either)
//! 3. If SAT, check if all integer variables have integer values
//! 4. If all integers are satisfied, return SAT
//! 5. Otherwise, try cutting planes (Gomory, then HNF)
//! 6. If no cuts, return a split request for branch-and-bound
//!
//! ## Cutting Planes
//!
//! - **Gomory cuts**: Derived from the simplex tableau. Fast but limited when
//!   the tableau involves slack variables (internal to simplex).
//! - **HNF cuts**: Derived from the original constraint matrix using Hermite
//!   Normal Form. Works even when Gomory cuts fail due to slack variables.
//!
//! The DPLL(T) framework handles the branching by backtracking on the conflict
//! and trying alternative Boolean assignments.

#![warn(missing_docs)]
#![warn(clippy::all)]
// LIA uses complex types for representing polynomial substitutions
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

mod dioph;
mod hnf;

use hashbrown::{HashMap, HashSet};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use z4_core::term::{Constant, Symbol, TermData, TermId, TermStore};
use z4_core::{Sort, SplitRequest, TheoryLit, TheoryPropagation, TheoryResult, TheorySolver};
use z4_lra::LraSolver;

#[derive(Clone, Copy, Debug)]
enum IneqOp {
    Ge,
    Le,
    Gt,
    Lt,
}

/// Model extracted from LIA solver with variable assignments
#[derive(Debug, Clone)]
pub struct LiaModel {
    /// Variable assignments: term_id -> integer value
    pub values: HashMap<TermId, BigInt>,
}

/// A stored HNF cut using TermIds (stable across LRA resets)
#[derive(Clone)]
pub struct StoredCut {
    /// Coefficients: term_id -> coefficient
    coeffs: Vec<(TermId, BigRational)>,
    /// The bound value
    bound: BigRational,
    /// True if lower bound (>= bound), false if upper bound (<= bound)
    is_lower: bool,
}

/// LIA theory solver using Gomory cuts, HNF cuts, and branch-and-bound over LRA
pub struct LiaSolver<'a> {
    /// Reference to the term store for parsing expressions
    terms: &'a TermStore,
    /// Underlying LRA solver for the relaxation
    lra: LraSolver<'a>,
    /// Set of term IDs known to be integer variables
    integer_vars: HashSet<TermId>,
    /// Asserted atoms for conflict generation
    asserted: Vec<(TermId, bool)>,
    /// Scope markers for push/pop
    scopes: Vec<usize>,
    /// Number of Gomory cut iterations attempted
    gomory_iterations: usize,
    /// Maximum Gomory cut iterations before falling back to split
    max_gomory_iterations: usize,
    /// Number of HNF cut iterations attempted
    hnf_iterations: usize,
    /// Maximum HNF cut iterations
    max_hnf_iterations: usize,
    /// Deduplicate HNF cuts across the solve (cuts are globally valid).
    seen_hnf_cuts: HashSet<HnfCutKey>,
    /// Stored cuts using TermIds for replay after LRA reset.
    /// These are derived from equality constraints and should be valid
    /// across different SAT models with the same base constraints.
    learned_cuts: Vec<StoredCut>,
    /// Cached set of asserted equality atoms (used to avoid re-running Diophantine
    /// solving when only inequalities change due to branching).
    /// Diophantine solving is skipped if this matches the current equality atoms.
    dioph_equality_key: Vec<TermId>,
    /// Integer variables that are provably dependent on other integer variables
    /// via unit-coefficient equalities (safe substitutions).
    ///
    /// These are typically poor branching candidates because their integrality
    /// is implied by other variables.
    dioph_safe_dependent_vars: HashSet<TermId>,
    /// Cached substitutions from Diophantine solver for bound propagation.
    /// Format: (substituted_term, [(dep_term, coeff)...], constant)
    /// Meaning: substituted_term = constant + Σ(coeff * dep_term)
    dioph_cached_substitutions: Vec<(TermId, Vec<(TermId, BigInt)>, BigInt)>,
}

/// Key for deduplicating HNF cuts (uses TermIds for stability across theory instances)
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct HnfCutKey {
    coeffs: Vec<(TermId, BigInt)>,
    bound: BigInt,
}

/// Diophantine solver state for external storage.
///
/// Contains: (equality_key, safe_dependent_vars, cached_substitutions)
/// - `equality_key`: List of asserted equality atoms (used to detect when re-analysis is needed)
/// - `safe_dependent_vars`: Variables that are poor branching candidates (dependent on others)
/// - `cached_substitutions`: Variable elimination expressions for bound propagation
pub type DiophState = (
    Vec<TermId>,                                  // equality_key
    HashSet<TermId>,                              // safe_dependent_vars
    Vec<(TermId, Vec<(TermId, BigInt)>, BigInt)>, // cached_substitutions
);

impl<'a> LiaSolver<'a> {
    /// Create a new LIA solver
    #[must_use]
    pub fn new(terms: &'a TermStore) -> Self {
        LiaSolver {
            terms,
            lra: LraSolver::new(terms),
            integer_vars: HashSet::new(),
            asserted: Vec::new(),
            scopes: Vec::new(),
            gomory_iterations: 0,
            max_gomory_iterations: 100, // Limit iterations to prevent infinite loops
            hnf_iterations: 0,
            max_hnf_iterations: 50, // HNF is more expensive, limit more
            seen_hnf_cuts: HashSet::new(),
            learned_cuts: Vec::new(),
            dioph_equality_key: Vec::new(),
            dioph_safe_dependent_vars: HashSet::new(),
            dioph_cached_substitutions: Vec::new(),
        }
    }

    /// Register a term as an integer variable
    ///
    /// Should be called for all variables declared with Int sort.
    pub fn register_integer_var(&mut self, term: TermId) {
        self.integer_vars.insert(term);
    }

    /// Check if a rational value is an integer
    fn is_integer(val: &BigRational) -> bool {
        val.denom().is_one()
    }

    /// Compute floor/ceil for an exact rational with correct behavior for negatives.
    fn floor_ceil_rational(value: &BigRational) -> (BigInt, BigInt) {
        let numer = value.numer();
        let denom = value.denom(); // BigRational normalizes denom > 0
        let q = numer / denom;
        let r = numer % denom;

        if r.is_zero() {
            (q.clone(), q)
        } else if numer.is_negative() {
            (&q - BigInt::one(), q)
        } else {
            (q.clone(), q + BigInt::one())
        }
    }

    /// Detect immediate integer infeasibility from bounds alone.
    ///
    /// For integer variables, strict/real bounds can imply a tightened integer interval.
    /// Example: `x > 5` and `x < 6` with `x : Int` is immediately UNSAT.
    fn check_integer_bounds_conflict(&self) -> Option<Vec<TheoryLit>> {
        // Sort integer vars for deterministic iteration order
        let mut int_vars: Vec<_> = self.integer_vars.iter().copied().collect();
        int_vars.sort_by_key(|t| t.0);

        for term in int_vars {
            let Some((lower, upper)) = self.lra.get_bounds(term) else {
                continue;
            };

            let lower_int = lower.as_ref().map(|b| {
                let (_floor, mut ceil) = Self::floor_ceil_rational(&b.value);
                if b.strict && Self::is_integer(&b.value) {
                    ceil += BigInt::one();
                }
                ceil
            });

            let upper_int = upper.as_ref().map(|b| {
                let (mut floor, _ceil) = Self::floor_ceil_rational(&b.value);
                if b.strict && Self::is_integer(&b.value) {
                    floor -= BigInt::one();
                }
                floor
            });

            if let (Some(li), Some(ui)) = (lower_int, upper_int) {
                if li > ui {
                    let mut reasons = Vec::new();
                    if let Some(lb) = lower {
                        reasons.push(TheoryLit::new(lb.reason, lb.reason_value));
                    }
                    if let Some(ub) = upper {
                        reasons.push(TheoryLit::new(ub.reason, ub.reason_value));
                    }
                    return Some(reasons);
                }
            }
        }

        None
    }

    /// Extract integer variables from a term and its subterms
    fn collect_integer_vars(&mut self, term: TermId) {
        match self.terms.get(term) {
            TermData::Var(_, _) => {
                // Check the sort of this term to see if it's an integer
                if matches!(self.terms.sort(term), Sort::Int) {
                    self.integer_vars.insert(term);
                }
            }
            TermData::App(_, args) => {
                for &arg in args {
                    self.collect_integer_vars(arg);
                }
            }
            TermData::Let(_, body) => {
                self.collect_integer_vars(*body);
            }
            TermData::Not(inner) => {
                self.collect_integer_vars(*inner);
            }
            TermData::Ite(cond, then_branch, else_branch) => {
                self.collect_integer_vars(*cond);
                self.collect_integer_vars(*then_branch);
                self.collect_integer_vars(*else_branch);
            }
            _ => {}
        }
    }

    /// Compute fractional part score for branching priority.
    /// Returns min(frac, 1-frac) where frac = value - floor(value).
    /// Smaller score = value is closer to an integer = lower priority for branching.
    /// Variables with score ~0.5 are furthest from any integer (best for branching).
    fn fractional_score(value: &BigRational) -> BigRational {
        let floor_val = Self::floor_ceil_rational(value).0;
        let frac = value - BigRational::from(floor_val);
        let half = BigRational::new(BigInt::one(), BigInt::from(2));
        if frac <= half {
            frac
        } else {
            BigRational::one() - &frac
        }
    }

    /// Compute bound span for a variable. Returns None if unbounded.
    fn bound_span(&self, term: TermId) -> Option<BigRational> {
        if let Some((Some(lower), Some(upper))) = self.lra.get_bounds(term) {
            Some(&upper.value - &lower.value)
        } else {
            None
        }
    }

    /// Check if a variable is "boxed" (has both lower and upper bounds).
    fn is_boxed(&self, term: TermId) -> bool {
        matches!(self.lra.get_bounds(term), Some((Some(_), Some(_))))
    }

    /// Select the best integer-infeasible variable for branching.
    /// Uses Z3-style heuristics:
    /// 1. Prefer boxed variables with small span
    /// 2. Prefer variables with values close to integers (stable split)
    /// 3. Use random tie-breaking for diversity
    ///
    /// Returns the selected variable and its value, or None if all satisfied.
    fn check_integer_constraints(&self) -> Option<(TermId, BigRational)> {
        let model = self.lra.extract_model();
        let debug = std::env::var("Z4_DEBUG_LIA_BRANCH").is_ok();

        // Collect all fractional integer variables with their values
        let mut candidates: Vec<(TermId, BigRational)> = Vec::new();
        for &term in &self.integer_vars {
            if let Some(val) = model.values.get(&term) {
                if !Self::is_integer(val) {
                    candidates.push((term, val.clone()));
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // Prefer branching on variables that are not already known to be dependent
        // via unit-coefficient equalities (e.g. v2 = 4*v1 + 5*v3 + 4).
        let mut independent: Vec<(TermId, BigRational)> = Vec::new();
        let mut dependent: Vec<(TermId, BigRational)> = Vec::new();
        for (term, value) in candidates {
            if self.dioph_safe_dependent_vars.contains(&term) {
                dependent.push((term, value));
            } else {
                independent.push((term, value));
            }
        }
        let candidates = if independent.is_empty() {
            dependent
        } else {
            independent
        };

        if debug {
            eprintln!(
                "[LIA] {} fractional integer variables found",
                candidates.len()
            );
        }

        // Categorize candidates by priority
        // Category 1: Boxed with small span (<= 1024)
        // Category 2: Small value or close to a bound
        // Category 3: Everything else
        let small_span_threshold = BigRational::from(BigInt::from(1024));

        let mut boxed_small: Vec<(TermId, BigRational, BigRational)> = Vec::new(); // (term, value, score)
        let mut others: Vec<(TermId, BigRational, BigRational)> = Vec::new();

        for (term, value) in candidates {
            let score = Self::fractional_score(&value);

            if self.is_boxed(term) {
                if let Some(span) = self.bound_span(term) {
                    if span <= small_span_threshold {
                        boxed_small.push((term, value, score));
                        continue;
                    }
                }
            }
            others.push((term, value, score));
        }

        // Sort each category by score (lower = closer to integer = prefer first for stability)
        // Z3 prefers variables closer to integers as they produce tighter cuts.
        // Use TermId as tie-breaker for deterministic behavior.
        boxed_small.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0 .0.cmp(&b.0 .0)));
        others.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0 .0.cmp(&b.0 .0)));

        // Select from highest priority non-empty category
        let selected = if !boxed_small.is_empty() {
            if debug {
                eprintln!(
                    "[LIA] Selecting from {} boxed-small candidates",
                    boxed_small.len()
                );
            }
            // Pick first (best score) from boxed_small
            let (term, value, score) = boxed_small.remove(0);
            if debug {
                eprintln!(
                    "[LIA] Selected term {} with value {}, score {}",
                    term.0, value, score
                );
            }
            (term, value)
        } else {
            if debug {
                eprintln!("[LIA] Selecting from {} other candidates", others.len());
            }
            // Pick first (best score) from others
            let (term, value, score) = others.remove(0);
            if debug {
                eprintln!(
                    "[LIA] Selected term {} with value {}, score {}",
                    term.0, value, score
                );
            }
            (term, value)
        };

        Some(selected)
    }

    /// Create a split request for branch-and-bound.
    ///
    /// When variable x has value v (non-integer), we request a split:
    /// (x <= floor(v)) OR (x >= ceil(v))
    fn create_split_request(&self, variable: TermId, value: BigRational) -> SplitRequest {
        let (floor, ceil) = Self::floor_ceil_rational(&value);

        SplitRequest {
            variable,
            value,
            floor,
            ceil,
        }
    }

    /// Try patching fractional integer variables to avoid branching.
    ///
    /// This is Z3's "patching" technique that adjusts non-basic integer variables
    /// to make fractional basic variables integral. It avoids branching entirely
    /// when successful, which can dramatically speed up solving.
    ///
    /// Returns true if any patching succeeded (caller should re-check).
    fn try_patching(&mut self) -> bool {
        let debug = std::env::var("Z4_DEBUG_PATCH").is_ok();
        let model = self.lra.extract_model();

        // Find all fractional integer variables
        let mut fractional_vars: Vec<TermId> = Vec::new();
        for &term in &self.integer_vars {
            if let Some(val) = model.values.get(&term) {
                if !Self::is_integer(val) {
                    fractional_vars.push(term);
                }
            }
        }

        if fractional_vars.is_empty() {
            return false;
        }

        if debug {
            eprintln!(
                "[PATCH] Trying to patch {} fractional vars",
                fractional_vars.len()
            );
        }

        // Try patching each fractional variable
        for term in fractional_vars {
            if self.lra.try_patch_integer_var(term, &self.integer_vars) {
                // Patching one variable may fix multiple, so return immediately
                // to re-check the state
                return true;
            }
        }

        false
    }

    /// Extract the current model if satisfiable
    ///
    /// Returns None if the last check was not SAT or if integer constraints
    /// are not satisfied.
    pub fn extract_model(&self) -> Option<LiaModel> {
        let debug = std::env::var("Z4_DEBUG_LIA").is_ok();
        let lra_model = self.lra.extract_model();
        let mut values = HashMap::new();

        if debug {
            eprintln!(
                "[LIA] extract_model: lra_model has {} values, integer_vars has {} entries",
                lra_model.values.len(),
                self.integer_vars.len()
            );
            for &term in &self.integer_vars {
                eprintln!("[LIA] integer_var: term {}", term.0);
            }
        }

        // Convert rational values to integers, checking constraints
        for (&term, val) in &lra_model.values {
            if debug {
                eprintln!(
                    "[LIA] checking term {}: in integer_vars={}",
                    term.0,
                    self.integer_vars.contains(&term)
                );
            }
            if self.integer_vars.contains(&term) {
                if Self::is_integer(val) {
                    if debug {
                        eprintln!("[LIA] term {} -> int value {}", term.0, val.numer());
                    }
                    values.insert(term, val.numer().clone());
                } else {
                    // Integer constraint violated
                    if debug {
                        eprintln!("[LIA] term {} has non-integer value {}", term.0, val);
                    }
                    return None;
                }
            }
        }

        if debug {
            eprintln!("[LIA] final model has {} values", values.len());
        }
        Some(LiaModel { values })
    }

    /// Get the underlying LRA solver
    pub fn lra_solver(&self) -> &LraSolver<'a> {
        &self.lra
    }

    /// Count the number of equality constraints in the asserted literals.
    ///
    /// Used to detect equality-dense problems where more aggressive HNF
    /// cut generation is beneficial.
    fn count_equalities(&self) -> usize {
        let mut count = 0;
        for &(literal, value) in &self.asserted {
            if !value {
                continue;
            }

            if let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) {
                if name == "=" && args.len() == 2 {
                    count += 1;
                }
            }
        }
        count
    }

    /// Compute a stable key for the currently asserted equality atoms.
    ///
    /// Used to avoid re-running Diophantine solving when only inequalities change
    /// (common during branch-and-bound).
    fn equality_key(&self) -> Vec<TermId> {
        let mut key: Vec<TermId> = Vec::new();

        for &(literal, value) in &self.asserted {
            if !value {
                continue;
            }

            if let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) {
                if name == "=" && args.len() == 2 {
                    key.push(literal);
                }
            }
        }

        key.sort_by_key(|t| t.0);
        key.dedup();
        key
    }

    /// GCD test for integer feasibility.
    ///
    /// For an equation `Σ(a_i * x_i) = c` where all x_i are integers,
    /// if GCD(a_1, a_2, ..., a_n) does not divide c, the equation is UNSAT.
    ///
    /// Example: 4*x + 4*y + 4*z - 2*w = 49
    /// GCD(4, 4, 4, 2) = 2, but 2 does not divide 49, so UNSAT.
    fn gcd_test(&self) -> Option<Vec<TheoryLit>> {
        let debug = std::env::var("Z4_DEBUG_GCD").is_ok();

        if debug {
            eprintln!(
                "[GCD] Running GCD test on {} asserted literals",
                self.asserted.len()
            );
        }

        for &(literal, value) in &self.asserted {
            // Only test positive equality assertions
            if !value {
                continue;
            }

            // Check if this is an equality
            let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) else {
                continue;
            };
            if name != "=" || args.len() != 2 {
                continue;
            }

            // Parse both sides of the equality
            let (coeffs, constant) = self.parse_linear_expr_for_gcd(args[0], args[1]);

            if coeffs.is_empty() {
                continue;
            }

            if debug {
                eprintln!("[GCD] Equality: coeffs={:?}, constant={}", coeffs, constant);
            }

            // All coefficients must be integer (which they should be for LIA)
            // Compute GCD of all coefficients
            let mut gcd = BigInt::zero();
            for coeff in &coeffs {
                gcd = gcd.gcd(coeff);
            }

            if gcd.is_zero() {
                continue;
            }

            // Check if GCD divides the constant
            // The constant is on the RHS, so we're checking GCD | constant
            let remainder = &constant % &gcd;
            if !remainder.is_zero() {
                if debug {
                    eprintln!(
                        "[GCD] UNSAT: GCD={} does not divide constant={} (remainder={})",
                        gcd, constant, remainder
                    );
                }
                // Return conflict with the equality as the reason
                return Some(vec![TheoryLit::new(literal, true)]);
            }
        }

        None
    }

    /// Check modular constraints from single equalities against bounds.
    ///
    /// For an equality like `r = 2*x - 2*y`, if variable `r` has coefficient ±1
    /// and all other coefficients have GCD > 1, then `r ≡ constant (mod GCD)`.
    ///
    /// Combined with bounds on `r`, this can detect infeasibility.
    fn check_single_equality_modular_constraints(&self) -> Option<Vec<TheoryLit>> {
        let debug = std::env::var("Z4_DEBUG_MOD").is_ok();

        for &(literal, value) in &self.asserted {
            // Only test positive equality assertions
            if !value {
                continue;
            }

            // Check if this is an equality
            let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) else {
                continue;
            };
            if name != "=" || args.len() != 2 {
                continue;
            }

            // Parse the equality with variable tracking
            let (var_coeffs, constant) = self.parse_linear_expr_with_vars(args[0], args[1]);

            if var_coeffs.is_empty() {
                continue;
            }

            // Find variables with coefficient ±1
            for (&var_term, var_coeff) in &var_coeffs {
                if var_coeff.abs() != BigInt::one() {
                    continue;
                }

                // Compute GCD of all other coefficients
                let mut other_gcd = BigInt::zero();
                for (&other_term, other_coeff) in &var_coeffs {
                    if other_term == var_term {
                        continue;
                    }
                    if other_gcd.is_zero() {
                        other_gcd = other_coeff.abs();
                    } else {
                        other_gcd = other_gcd.gcd(&other_coeff.abs());
                    }
                    // Early exit if GCD becomes 1
                    if other_gcd.is_one() {
                        break;
                    }
                }

                // If other coefficients have GCD > 1, we have a modular constraint
                if other_gcd > BigInt::one() {
                    // var = constant/coeff + sum(other_i/coeff * x_i)
                    // Since coeff = ±1, var = ±constant ± sum(other_i * x_i)
                    // The sum is divisible by other_gcd, so var ≡ ±constant (mod other_gcd)
                    let residue = {
                        // If coeff = -1, negate constant
                        let adj_const = if var_coeff.is_negative() {
                            -&constant
                        } else {
                            constant.clone()
                        };
                        let r = &adj_const % &other_gcd;
                        if r < BigInt::zero() {
                            r + &other_gcd
                        } else {
                            r
                        }
                    };

                    if debug {
                        eprintln!(
                            "[MOD] From equality {:?}: {:?} ≡ {} (mod {})",
                            literal, var_term, residue, other_gcd
                        );
                    }

                    // Check bounds on var_term
                    if let Some((lb_opt, ub_opt)) = self.lra.get_bounds(var_term) {
                        let effective_lb = lb_opt.map(|b| Self::ceil_bigint(&b.value));
                        let effective_ub = ub_opt.map(|b| Self::floor_bigint(&b.value));

                        if let (Some(lb), Some(ub)) = (&effective_lb, &effective_ub) {
                            if debug {
                                eprintln!("[MOD] Variable {:?} bounds: [{}, {}]", var_term, lb, ub);
                            }

                            // Find first valid integer >= lb satisfying modular constraint
                            let diff = &residue - lb;
                            let adjustment = {
                                let r = &diff % &other_gcd;
                                if r < BigInt::zero() {
                                    r + &other_gcd
                                } else {
                                    r
                                }
                            };
                            let first_valid = lb + adjustment;

                            if &first_valid > ub {
                                if debug {
                                    eprintln!(
                                        "[MOD] UNSAT: no integer in [{}, {}] satisfies ≡ {} (mod {})",
                                        lb, ub, residue, other_gcd
                                    );
                                }
                                // Return conflict
                                return Some(vec![TheoryLit::new(literal, true)]);
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Check if a disequality split conflicts with modular constraints.
    ///
    /// When a disequality split is requested for variable V with excluded value E,
    /// check if modular constraints from equalities make E the only valid integer
    /// in V's bounds. If so, the disequality V != E makes the formula UNSAT.
    fn check_disequality_vs_modular(
        &self,
        split: &z4_core::DisequlitySplitRequest,
    ) -> Option<Vec<TheoryLit>> {
        let debug = std::env::var("Z4_DEBUG_MOD").is_ok();

        if debug {
            eprintln!(
                "[MOD] check_disequality_vs_modular: var={:?}, excluded={}",
                split.variable, split.excluded_value
            );
            eprintln!("[MOD] Asserted literals: {}", self.asserted.len());
        }

        let excluded_int = if split.excluded_value.is_integer() {
            split.excluded_value.numer().clone()
        } else {
            if debug {
                eprintln!("[MOD] Excluded value is not integer, skipping");
            }
            return None; // Non-integer excluded value
        };

        // Check each equality for modular constraints on split.variable
        // First, check Diophantine substitutions (most reliable source of modular constraints)
        if debug {
            eprintln!(
                "[MOD] dioph_cached_substitutions count: {}",
                self.dioph_cached_substitutions.len()
            );
        }
        for (term_id, coeffs, constant) in &self.dioph_cached_substitutions {
            if *term_id != split.variable {
                continue;
            }

            // Compute GCD of all coefficients in the substitution
            let mut gcd = BigInt::zero();
            for (_, coeff) in coeffs {
                if gcd.is_zero() {
                    gcd = coeff.abs();
                } else {
                    gcd = gcd.gcd(&coeff.abs());
                }
                if gcd.is_one() {
                    break;
                }
            }

            if gcd > BigInt::one() {
                // var ≡ constant (mod gcd)
                let residue = {
                    let r = constant % &gcd;
                    if r < BigInt::zero() {
                        r + &gcd
                    } else {
                        r.clone()
                    }
                };

                if debug {
                    eprintln!(
                        "[MOD] From dioph substitution: {:?} ≡ {} (mod {})",
                        term_id, residue, gcd
                    );
                }

                // Get bounds on split.variable
                if let Some((lb_opt, ub_opt)) = self.lra.get_bounds(split.variable) {
                    let effective_lb = lb_opt.map(|b| {
                        if b.strict && Self::is_integer(&b.value) {
                            Self::floor_bigint(&b.value) + BigInt::one()
                        } else {
                            Self::ceil_bigint(&b.value)
                        }
                    });
                    let effective_ub = ub_opt.map(|b| {
                        if b.strict && Self::is_integer(&b.value) {
                            Self::floor_bigint(&b.value) - BigInt::one()
                        } else {
                            Self::floor_bigint(&b.value)
                        }
                    });

                    if let (Some(lb), Some(ub)) = (&effective_lb, &effective_ub) {
                        let diff = &residue - lb;
                        let adjustment = {
                            let r = &diff % &gcd;
                            if r < BigInt::zero() {
                                r + &gcd
                            } else {
                                r
                            }
                        };
                        let first_valid = lb + adjustment;

                        if &first_valid <= ub {
                            let second_valid = &first_valid + &gcd;
                            if &second_valid > ub && first_valid == excluded_int {
                                if debug {
                                    eprintln!(
                                        "[MOD] UNSAT from dioph: unique value {} excluded for {:?}",
                                        excluded_int, split.variable
                                    );
                                }
                                // Return conflict with all positive assertions
                                let conflict: Vec<TheoryLit> = self
                                    .asserted
                                    .iter()
                                    .map(|&(lit, val)| TheoryLit::new(lit, val))
                                    .collect();
                                return Some(conflict);
                            }
                        }
                    }
                }
            }
        }

        // Also check single equalities for modular constraints
        for &(literal, value) in &self.asserted {
            if !value {
                continue;
            }

            let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) else {
                continue;
            };
            if name != "=" || args.len() != 2 {
                continue;
            }

            // Parse the equality
            let (var_coeffs, constant) = self.parse_linear_expr_with_vars(args[0], args[1]);

            if debug {
                eprintln!(
                    "[MOD] Equality {:?}: {} variables, constant={}",
                    literal,
                    var_coeffs.len(),
                    constant
                );
                for (var, coeff) in &var_coeffs {
                    eprintln!("[MOD]   {:?} -> coeff {}", var, coeff);
                }
            }

            // Check if split.variable appears with coefficient ±1
            let Some(var_coeff) = var_coeffs.get(&split.variable) else {
                if debug {
                    eprintln!(
                        "[MOD]   split.variable {:?} not found in equality",
                        split.variable
                    );
                }
                continue;
            };
            if var_coeff.abs() != BigInt::one() {
                continue;
            }

            // Compute GCD of other coefficients
            let mut other_gcd = BigInt::zero();
            for (&other_term, other_coeff) in &var_coeffs {
                if other_term == split.variable {
                    continue;
                }
                if other_gcd.is_zero() {
                    other_gcd = other_coeff.abs();
                } else {
                    other_gcd = other_gcd.gcd(&other_coeff.abs());
                }
                if other_gcd.is_one() {
                    break;
                }
            }

            if debug {
                eprintln!("[MOD]   var_coeff={}, other_gcd={}", var_coeff, other_gcd);
            }

            if other_gcd <= BigInt::one() {
                if debug {
                    eprintln!("[MOD]   other_gcd <= 1, skipping");
                }
                continue;
            }

            // Compute residue for the modular constraint
            let residue = {
                let adj_const = if var_coeff.is_negative() {
                    -&constant
                } else {
                    constant.clone()
                };
                let r = &adj_const % &other_gcd;
                if r < BigInt::zero() {
                    r + &other_gcd
                } else {
                    r
                }
            };

            if debug {
                eprintln!("[MOD]   residue={}", residue);
            }

            // Get bounds on split.variable
            if let Some((lb_opt, ub_opt)) = self.lra.get_bounds(split.variable) {
                if debug {
                    eprintln!("[MOD]   bounds: lb={:?}, ub={:?}", lb_opt, ub_opt);
                }
                // For integers, strict bounds need adjustment:
                // lb > v means lb >= ceil(v) for strict, lb >= ceil(v) for non-strict
                // ub < v means ub <= floor(v)-1 for strict integer if v is integer
                let effective_lb = lb_opt.map(|b| {
                    if b.strict && Self::is_integer(&b.value) {
                        // strict: x > n means x >= n+1 for integers
                        Self::floor_bigint(&b.value) + BigInt::one()
                    } else {
                        Self::ceil_bigint(&b.value)
                    }
                });
                let effective_ub = ub_opt.map(|b| {
                    if b.strict && Self::is_integer(&b.value) {
                        // strict: x < n means x <= n-1 for integers
                        Self::floor_bigint(&b.value) - BigInt::one()
                    } else {
                        Self::floor_bigint(&b.value)
                    }
                });

                if let (Some(lb), Some(ub)) = (&effective_lb, &effective_ub) {
                    if debug {
                        eprintln!("[MOD]   effective bounds: [{}, {}]", lb, ub);
                    }

                    // Find first valid integer in [lb, ub] satisfying modular constraint
                    let diff = &residue - lb;
                    let adjustment = {
                        let r = &diff % &other_gcd;
                        if r < BigInt::zero() {
                            r + &other_gcd
                        } else {
                            r
                        }
                    };
                    let first_valid = lb + adjustment;

                    if debug {
                        eprintln!(
                            "[MOD]   first_valid={}, excluded_int={}",
                            first_valid, excluded_int
                        );
                    }

                    // Check if first_valid is the only valid integer and equals excluded_int
                    if &first_valid <= ub {
                        let second_valid = &first_valid + &other_gcd;
                        if debug {
                            eprintln!("[MOD]   second_valid={}, checking if second > ub ({}) and first == excluded",
                                      second_valid, ub);
                        }
                        if &second_valid > ub && first_valid == excluded_int {
                            // The excluded value is the ONLY valid integer!
                            if debug {
                                eprintln!(
                                    "[MOD] Disequality excludes unique valid value {} for {:?}",
                                    excluded_int, split.variable
                                );
                                eprintln!(
                                    "[MOD] Bounds [{}, {}], residue {} (mod {})",
                                    lb, ub, residue, other_gcd
                                );
                            }
                            // Return conflict with the equality and any disequality literals
                            let mut conflict = vec![TheoryLit::new(literal, true)];
                            // Add any asserted disequality for this variable
                            for &(diseq_lit, diseq_val) in &self.asserted {
                                if diseq_val {
                                    continue; // We want negated equalities (disequalities)
                                }
                                if let TermData::App(Symbol::Named(n), a) =
                                    self.terms.get(diseq_lit)
                                {
                                    if n == "=" && a.len() == 2 {
                                        // Check if this is the disequality for our variable
                                        let has_var = a.contains(&split.variable);
                                        if has_var {
                                            conflict.push(TheoryLit::new(diseq_lit, false));
                                        }
                                    }
                                }
                            }
                            return Some(conflict);
                        }
                    }
                }
            }
        }

        None
    }

    /// Parse a linear expression with variable tracking.
    /// Returns (variable -> coefficient map, constant) for equation Σ(coeff * var) = constant.
    fn parse_linear_expr_with_vars(
        &self,
        lhs: TermId,
        rhs: TermId,
    ) -> (HashMap<TermId, BigInt>, BigInt) {
        let mut var_coeffs: HashMap<TermId, BigInt> = HashMap::new();
        let mut constant = BigInt::zero();

        // Parse lhs with positive sign
        self.collect_linear_terms_with_vars(lhs, &BigInt::one(), &mut var_coeffs, &mut constant);

        // Parse rhs with negative sign
        self.collect_linear_terms_with_vars(rhs, &-BigInt::one(), &mut var_coeffs, &mut constant);

        // Negate constant (we're solving for Σ(coeff * var) = constant)
        constant = -constant;

        (var_coeffs, constant)
    }

    /// Recursively collect linear terms with variable tracking.
    fn collect_linear_terms_with_vars(
        &self,
        term: TermId,
        scale: &BigInt,
        var_coeffs: &mut HashMap<TermId, BigInt>,
        constant: &mut BigInt,
    ) {
        match self.terms.get(term) {
            TermData::Const(Constant::Int(n)) => {
                *constant += scale * n;
            }
            TermData::Const(Constant::Rational(r)) => {
                if r.0.denom().is_one() {
                    *constant += scale * r.0.numer();
                }
            }
            TermData::Var(_, _) => {
                *var_coeffs.entry(term).or_insert_with(BigInt::zero) += scale;
            }
            TermData::App(Symbol::Named(name), args) => match name.as_str() {
                "+" => {
                    for &arg in args {
                        self.collect_linear_terms_with_vars(arg, scale, var_coeffs, constant);
                    }
                }
                "-" if args.len() == 1 => {
                    self.collect_linear_terms_with_vars(
                        args[0],
                        &-scale.clone(),
                        var_coeffs,
                        constant,
                    );
                }
                "-" if args.len() >= 2 => {
                    self.collect_linear_terms_with_vars(args[0], scale, var_coeffs, constant);
                    for &arg in &args[1..] {
                        self.collect_linear_terms_with_vars(
                            arg,
                            &-scale.clone(),
                            var_coeffs,
                            constant,
                        );
                    }
                }
                "*" => {
                    let mut const_factor = BigInt::one();
                    let mut var_args = Vec::new();

                    for &arg in args {
                        if let Some(c) = self.extract_constant(arg) {
                            const_factor *= c;
                        } else {
                            var_args.push(arg);
                        }
                    }

                    let new_scale = scale * &const_factor;

                    if var_args.is_empty() {
                        *constant += &new_scale;
                    } else if var_args.len() == 1 {
                        self.collect_linear_terms_with_vars(
                            var_args[0],
                            &new_scale,
                            var_coeffs,
                            constant,
                        );
                    } else {
                        // Non-linear: treat entire term as opaque variable
                        *var_coeffs.entry(term).or_insert_with(BigInt::zero) += scale;
                    }
                }
                _ => {
                    // Unknown function - treat as opaque variable
                    *var_coeffs.entry(term).or_insert_with(BigInt::zero) += scale;
                }
            },
            _ => {
                // Unknown term - treat as variable
                *var_coeffs.entry(term).or_insert_with(BigInt::zero) += scale;
            }
        }
    }

    /// Parse a linear expression from lhs - rhs for GCD test.
    /// Returns (coefficients, constant) where the equation is Σ(coeff * var) = constant.
    fn parse_linear_expr_for_gcd(&self, lhs: TermId, rhs: TermId) -> (Vec<BigInt>, BigInt) {
        let mut coeffs = Vec::new();
        let mut constant = BigInt::zero();

        // Parse lhs with positive sign
        self.collect_linear_terms(lhs, &BigInt::one(), &mut coeffs, &mut constant);

        // Parse rhs with negative sign (move to LHS)
        self.collect_linear_terms(rhs, &-BigInt::one(), &mut coeffs, &mut constant);

        // The equation is: Σ(coeff * var) - constant = 0
        // So: Σ(coeff * var) = constant
        // But we subtracted rhs, so constant is negative of what we want
        constant = -constant;

        (coeffs, constant)
    }

    /// Recursively collect linear terms from an expression.
    /// Adds coefficients to `coeffs` and accumulates constants.
    fn collect_linear_terms(
        &self,
        term: TermId,
        scale: &BigInt,
        coeffs: &mut Vec<BigInt>,
        constant: &mut BigInt,
    ) {
        match self.terms.get(term) {
            TermData::Const(Constant::Int(n)) => {
                *constant += scale * n;
            }
            TermData::Const(Constant::Rational(r)) => {
                // For LIA, rationals should have denominator 1
                if r.0.denom().is_one() {
                    *constant += scale * r.0.numer();
                }
                // Otherwise skip (non-integer constant)
            }
            TermData::Var(_, _) => {
                // Variable with coefficient = scale
                coeffs.push(scale.clone());
            }
            TermData::App(Symbol::Named(name), args) => {
                match name.as_str() {
                    "+" => {
                        for &arg in args {
                            self.collect_linear_terms(arg, scale, coeffs, constant);
                        }
                    }
                    "-" if args.len() == 1 => {
                        // Unary minus
                        self.collect_linear_terms(args[0], &-scale.clone(), coeffs, constant);
                    }
                    "-" if args.len() >= 2 => {
                        // Binary/n-ary minus
                        self.collect_linear_terms(args[0], scale, coeffs, constant);
                        for &arg in &args[1..] {
                            self.collect_linear_terms(arg, &-scale.clone(), coeffs, constant);
                        }
                    }
                    "*" => {
                        // Find constant factor and variable
                        let mut const_factor = BigInt::one();
                        let mut var_args = Vec::new();

                        for &arg in args {
                            if let Some(c) = self.extract_constant(arg) {
                                const_factor *= c;
                            } else {
                                var_args.push(arg);
                            }
                        }

                        let new_scale = scale * &const_factor;

                        if var_args.is_empty() {
                            // Pure constant
                            *constant += &new_scale;
                        } else if var_args.len() == 1 {
                            // Linear: const * var
                            self.collect_linear_terms(var_args[0], &new_scale, coeffs, constant);
                        }
                        // Non-linear terms are ignored for GCD test
                    }
                    _ => {
                        // Unknown function - treat as a single variable
                        coeffs.push(scale.clone());
                    }
                }
            }
            _ => {
                // Unknown term - treat as a variable
                coeffs.push(scale.clone());
            }
        }
    }

    /// Try solving a single 2-variable equality using extended GCD with bounds.
    ///
    /// For equations like `4*x + 3*y = 70` with bounds on x and y, use extended GCD
    /// to parameterize solutions and determine valid values directly.
    ///
    /// Returns Some(conflict) if infeasible, None if solving succeeded or wasn't applicable.
    fn try_two_variable_solve(&mut self) -> Option<Vec<TheoryLit>> {
        let debug = std::env::var("Z4_DEBUG_DIOPH").is_ok();

        // Count total equalities - we can only safely set hard bounds when there's
        // exactly one equality (otherwise bounds from one equation may conflict with others)
        let total_equalities = self.count_equalities();

        // Build term_to_idx mapping for all integer variables
        let mut term_to_idx: HashMap<TermId, usize> = HashMap::new();
        let mut idx_to_term: Vec<TermId> = Vec::new();

        let mut int_vars: Vec<TermId> = self.integer_vars.iter().copied().collect();
        int_vars.sort_by_key(|t| t.0);

        for (idx, term) in int_vars.into_iter().enumerate() {
            term_to_idx.insert(term, idx);
            idx_to_term.push(term);
        }

        // Collect candidates for 2-variable equations
        // We need to do this in two phases to avoid borrow conflicts
        struct TwoVarCandidate {
            literal: TermId,
            coeffs: HashMap<usize, BigInt>,
            constant: BigInt,
        }

        let mut candidates: Vec<TwoVarCandidate> = Vec::new();

        for &(literal, value) in &self.asserted {
            if !value {
                continue;
            }

            let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) else {
                continue;
            };

            if name != "=" || args.len() != 2 {
                continue;
            }

            // Parse the equality
            let (var_coeffs, constant) =
                self.parse_equality_for_dioph(args[0], args[1], &term_to_idx);

            // Must have exactly 2 variables
            if var_coeffs.len() != 2 {
                continue;
            }

            // Build the equation
            let mut coeffs_map: HashMap<usize, BigInt> = HashMap::new();
            for (idx, coeff) in var_coeffs {
                coeffs_map.insert(idx, coeff);
            }

            candidates.push(TwoVarCandidate {
                literal,
                coeffs: coeffs_map,
                constant,
            });
        }

        // Now process candidates (we've dropped the borrow on self.asserted)
        for candidate in candidates {
            let eq = dioph::IntEquation::new(candidate.coeffs, candidate.constant);

            // GCD infeasibility check
            if eq.gcd_infeasible() {
                if debug {
                    eprintln!("[DIOPH] 2-var GCD infeasibility");
                }
                return Some(vec![TheoryLit::new(candidate.literal, true)]);
            }

            // Try extended GCD solution
            let Some(sol) = eq.solve_two_variable() else {
                continue;
            };

            if debug {
                eprintln!(
                    "[DIOPH] 2-var solution: var_x={}, var_y={}, x0={}, y0={}, step_x={}, step_y={}",
                    sol.var_x, sol.var_y, sol.x0, sol.y0, sol.step_x, sol.step_y
                );
            }

            // Get bounds for both variables from LRA
            let term_x = idx_to_term.get(sol.var_x).copied();
            let term_y = idx_to_term.get(sol.var_y).copied();

            let (x_lo, x_hi) = self.get_integer_bounds_for_term(term_x);
            let (y_lo, y_hi) = self.get_integer_bounds_for_term(term_y);

            if debug {
                eprintln!(
                    "[DIOPH] Bounds: x in [{:?}, {:?}], y in [{:?}, {:?}]",
                    x_lo, x_hi, y_lo, y_hi
                );
            }

            // Compute k bounds from x bounds
            let (k_min_x, k_max_x) = sol.k_bounds_from_x(x_lo.as_ref(), x_hi.as_ref());
            // Compute k bounds from y bounds
            let (k_min_y, k_max_y) = sol.k_bounds_from_y(y_lo.as_ref(), y_hi.as_ref());

            // Intersect k bounds
            let k_min = match (k_min_x, k_min_y) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };
            let k_max = match (k_max_x, k_max_y) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };

            if debug {
                eprintln!("[DIOPH] k bounds: [{:?}, {:?}]", k_min, k_max);
            }

            // Check if bounds are contradictory
            if let (Some(ref lo), Some(ref hi)) = (&k_min, &k_max) {
                if lo > hi {
                    if debug {
                        eprintln!("[DIOPH] k bounds contradictory - infeasible");
                    }
                    // Infeasible - gather all bound assertions
                    let mut conflict = vec![TheoryLit::new(candidate.literal, true)];
                    // Add bounds that contributed to the conflict
                    conflict.extend(self.get_bound_reasons_for_term(term_x));
                    conflict.extend(self.get_bound_reasons_for_term(term_y));
                    return Some(conflict);
                }

                // If k has a small range, enumerate and set tighter bounds
                // IMPORTANT: Only do this when there's exactly one equality in the system.
                // If there are multiple equalities, setting hard bounds based on one equation
                // may conflict with constraints from other equations.
                let range = hi - lo;
                if range <= BigInt::from(10) && total_equalities == 1 {
                    // Pick the first valid k and set bounds
                    let k = lo.clone();
                    let (x_val, y_val) = sol.evaluate(&k);

                    if debug {
                        eprintln!("[DIOPH] Setting bounds: x={}, y={}", x_val, y_val);
                    }

                    if let Some(tid) = term_x {
                        self.add_integer_bound(tid, &x_val);
                    }
                    if let Some(tid) = term_y {
                        self.add_integer_bound(tid, &y_val);
                    }
                }
            }
        }

        None
    }

    /// Get integer bounds for a term by scanning asserted literals.
    ///
    /// This looks for constraints of the form `x >= c`, `x > c`, `x <= c`, `x < c`
    /// where x is the target term and c is a constant.
    fn get_integer_bounds_for_term(
        &self,
        term: Option<TermId>,
    ) -> (Option<BigInt>, Option<BigInt>) {
        let Some(tid) = term else {
            return (None, None);
        };

        let mut lower: Option<BigInt> = None;
        let mut upper: Option<BigInt> = None;

        for &(literal, value) in &self.asserted {
            if !value {
                continue;
            }

            let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) else {
                continue;
            };

            if args.len() != 2 {
                continue;
            }

            // Check if this is a bound on our target term
            // Pattern: tid OP constant or constant OP tid
            let (target, constant, is_target_on_left) = if args[0] == tid {
                // tid OP constant
                let const_val = self.extract_constant(args[1]);
                (tid, const_val, true)
            } else if args[1] == tid {
                // constant OP tid
                let const_val = self.extract_constant(args[0]);
                (tid, const_val, false)
            } else {
                continue;
            };

            let Some(c) = constant else {
                continue;
            };

            if target != tid {
                continue;
            }

            // Extract the bound based on the operator and which side the term is on
            match (name.as_str(), is_target_on_left) {
                (">=", true) => {
                    // tid >= c => lower bound c
                    lower = Some(lower.map_or(c.clone(), |l| l.max(c.clone())));
                }
                (">=", false) => {
                    // c >= tid => upper bound c
                    upper = Some(upper.map_or(c.clone(), |u| u.min(c.clone())));
                }
                (">", true) => {
                    // tid > c => lower bound c + 1 (for integers)
                    let bound = &c + BigInt::one();
                    lower = Some(lower.map_or(bound.clone(), |l| l.max(bound)));
                }
                (">", false) => {
                    // c > tid => upper bound c - 1 (for integers)
                    let bound = &c - BigInt::one();
                    upper = Some(upper.map_or(bound.clone(), |u| u.min(bound)));
                }
                ("<=", true) => {
                    // tid <= c => upper bound c
                    upper = Some(upper.map_or(c.clone(), |u| u.min(c.clone())));
                }
                ("<=", false) => {
                    // c <= tid => lower bound c
                    lower = Some(lower.map_or(c.clone(), |l| l.max(c.clone())));
                }
                ("<", true) => {
                    // tid < c => upper bound c - 1 (for integers)
                    let bound = &c - BigInt::one();
                    upper = Some(upper.map_or(bound.clone(), |u| u.min(bound)));
                }
                ("<", false) => {
                    // c < tid => lower bound c + 1 (for integers)
                    let bound = &c + BigInt::one();
                    lower = Some(lower.map_or(bound.clone(), |l| l.max(bound)));
                }
                _ => {}
            }
        }

        (lower, upper)
    }

    /// Extract an integer constant from a term, if it is one.
    fn extract_constant(&self, term: TermId) -> Option<BigInt> {
        match self.terms.get(term) {
            TermData::Const(Constant::Int(n)) => Some(n.clone()),
            TermData::Const(Constant::Rational(r)) => {
                if r.0.denom().is_one() {
                    Some(r.0.numer().clone())
                } else {
                    None
                }
            }
            TermData::App(Symbol::Named(name), args) if name == "-" && args.len() == 1 => {
                // Negation
                self.extract_constant(args[0]).map(|n| -n)
            }
            _ => None,
        }
    }

    /// Extract a rational constant from a term, if it is one.
    ///
    /// This is used when collecting rational-valued constraints for cut generation.
    fn extract_rational_constant(&self, term: TermId) -> Option<BigRational> {
        match self.terms.get(term) {
            TermData::Const(Constant::Int(n)) => Some(BigRational::from(n.clone())),
            TermData::Const(Constant::Rational(r)) => Some(r.0.clone()),
            TermData::App(Symbol::Named(name), args) if name == "-" && args.len() == 1 => {
                self.extract_rational_constant(args[0]).map(|n| -n)
            }
            _ => None,
        }
    }

    /// Get bound reason literals for a term (for conflict generation).
    fn get_bound_reasons_for_term(&self, term: Option<TermId>) -> Vec<TheoryLit> {
        let mut reasons = Vec::new();
        let Some(tid) = term else {
            return reasons;
        };

        // Find bound assertions for this term
        for &(literal, value) in &self.asserted {
            if !value {
                continue;
            }

            let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) else {
                continue;
            };

            if args.len() != 2 {
                continue;
            }

            // Check if this involves our target term as a simple bound
            let is_bound = (args[0] == tid || args[1] == tid)
                && matches!(name.as_str(), ">=" | "<=" | ">" | "<");

            if is_bound {
                // Verify one side is a constant
                let const_side = if args[0] == tid { args[1] } else { args[0] };
                if self.extract_constant(const_side).is_some() {
                    reasons.push(TheoryLit::new(literal, true));
                }
            }
        }

        reasons
    }

    /// Try solving the integer equality system directly using variable elimination.
    ///
    /// For equality-dense problems (k equalities with n variables, k close to n),
    /// this can directly determine variable values or detect infeasibility,
    /// avoiding the slow iterative HNF cuts approach.
    ///
    /// Returns Some(conflict) if infeasible, None if solving succeeded or wasn't applicable.
    fn try_diophantine_solve(&mut self) -> Option<Vec<TheoryLit>> {
        let debug = std::env::var("Z4_DEBUG_DIOPH").is_ok();
        self.dioph_safe_dependent_vars.clear();

        let num_equalities = self.count_equalities();
        let num_vars = self.integer_vars.len();

        // Only apply to equality-dense problems (at least 2 equalities)
        if num_equalities < 2 || num_vars == 0 {
            return None;
        }

        if debug {
            eprintln!(
                "[DIOPH] Trying Diophantine solve: {} equalities, {} vars",
                num_equalities, num_vars
            );
        }

        // Build term_to_idx mapping
        let mut term_to_idx: HashMap<TermId, usize> = HashMap::new();
        let mut idx_to_term: Vec<TermId> = Vec::new();

        let mut int_vars: Vec<TermId> = self.integer_vars.iter().copied().collect();
        int_vars.sort_by_key(|t| t.0);

        for (idx, term) in int_vars.into_iter().enumerate() {
            term_to_idx.insert(term, idx);
            idx_to_term.push(term);
        }

        // Collect all equalities
        let mut solver = dioph::DiophSolver::new();
        let mut equality_literals: Vec<TermId> = Vec::new();

        for &(literal, value) in &self.asserted {
            if !value {
                continue;
            }

            let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) else {
                continue;
            };

            if name != "=" || args.len() != 2 {
                continue;
            }

            // Parse the equality
            let (var_coeffs, constant) =
                self.parse_equality_for_dioph(args[0], args[1], &term_to_idx);

            if var_coeffs.is_empty() && !constant.is_zero() {
                // 0 = c where c != 0 - immediate infeasibility
                if debug {
                    eprintln!("[DIOPH] Immediate infeasibility: 0 = {}", constant);
                }
                return Some(vec![TheoryLit::new(literal, true)]);
            }

            if !var_coeffs.is_empty() {
                solver.add_equation_from(var_coeffs, constant);
                equality_literals.push(literal);
            }
        }

        if equality_literals.len() < 2 {
            return None; // Not enough equalities
        }

        // Try to solve
        let result = solver.solve();
        for var_idx in solver.safe_original_dependents(idx_to_term.len()) {
            if let Some(&term_id) = idx_to_term.get(var_idx) {
                self.dioph_safe_dependent_vars.insert(term_id);
            }
        }

        match result {
            dioph::DiophResult::Infeasible => {
                if debug {
                    eprintln!("[DIOPH] Infeasible - returning conflict with all equalities");
                }
                // Return all equality literals as the conflict
                let conflict: Vec<TheoryLit> = equality_literals
                    .iter()
                    .map(|&lit| TheoryLit::new(lit, true))
                    .collect();
                return Some(conflict);
            }
            dioph::DiophResult::Solved(values) => {
                if debug {
                    eprintln!("[DIOPH] Fully solved: {:?}", values);
                }
                // Add determined values as tight bounds to LRA
                for (var_idx, value) in values {
                    if var_idx < idx_to_term.len() {
                        let term_id = idx_to_term[var_idx];
                        self.add_integer_bound(term_id, &value);
                    }
                }
            }
            dioph::DiophResult::Partial { determined, .. } => {
                if debug {
                    eprintln!("[DIOPH] Partial solution: {:?}", determined);
                }
                // Add determined values as tight bounds
                for (var_idx, value) in determined {
                    if var_idx < idx_to_term.len() {
                        let term_id = idx_to_term[var_idx];
                        self.add_integer_bound(term_id, &value);
                    }
                }

                // Get substitution expressions for bound propagation
                // Convert to TermId format and cache for reuse
                let raw_subs = solver.get_substitutions_for_propagation(idx_to_term.len());
                self.dioph_cached_substitutions = raw_subs
                    .into_iter()
                    .filter_map(|(var_idx, coeffs, constant)| {
                        if var_idx >= idx_to_term.len() {
                            return None;
                        }
                        let term_id = idx_to_term[var_idx];
                        let term_coeffs: Vec<_> = coeffs
                            .into_iter()
                            .filter_map(|(dep_idx, c)| {
                                if dep_idx >= idx_to_term.len() {
                                    None
                                } else {
                                    Some((idx_to_term[dep_idx], c))
                                }
                            })
                            .collect();
                        // Only include if all deps are valid
                        if !term_coeffs.is_empty() {
                            Some((term_id, term_coeffs, constant))
                        } else {
                            None
                        }
                    })
                    .collect();

                if debug && !self.dioph_cached_substitutions.is_empty() {
                    eprintln!(
                        "[DIOPH] Cached {} substitutions for bound propagation",
                        self.dioph_cached_substitutions.len()
                    );
                }
            }
        }

        // Always propagate bounds through cached substitutions (even when structure is cached)
        self.propagate_bounds_through_substitutions();

        None
    }

    /// Try direct lattice enumeration for equality-dense systems.
    ///
    /// When we have k equalities in n variables with k >= n-2 (at most 2 free variables),
    /// use rational Gaussian elimination to express dependent vars as functions of
    /// free vars, then enumerate valid integer solutions directly.
    ///
    /// This avoids the 28+ branch-and-bound iterations that slow down system_05.
    fn try_direct_enumeration(&mut self) -> Option<Vec<TheoryLit>> {
        let debug = std::env::var("Z4_DEBUG_ENUM").is_ok();

        let num_equalities = self.count_equalities();
        let num_vars = self.integer_vars.len();

        // Only apply when we have at most 2 free variables
        // (k equalities reduce n vars to n-k free vars, assuming linear independence)
        if num_equalities == 0 || num_vars == 0 {
            return None;
        }
        let expected_free = num_vars.saturating_sub(num_equalities);
        if expected_free > 2 {
            return None; // Too many free vars for enumeration
        }

        if debug {
            eprintln!(
                "[ENUM] Trying direct enumeration: {} equalities, {} vars, ~{} free",
                num_equalities, num_vars, expected_free
            );
        }

        // Build term_to_idx mapping
        let mut term_to_idx: HashMap<TermId, usize> = HashMap::new();
        let mut idx_to_term: Vec<TermId> = Vec::new();

        let mut int_vars: Vec<TermId> = self.integer_vars.iter().copied().collect();
        int_vars.sort_by_key(|t| t.0);

        for (idx, term) in int_vars.into_iter().enumerate() {
            term_to_idx.insert(term, idx);
            idx_to_term.push(term);
        }

        // Build the rational matrix from equalities
        // Each row is [coeffs... | constant] representing sum(coeff_i * x_i) = constant
        let mut matrix: Vec<(Vec<BigRational>, BigRational)> = Vec::new();
        let mut equality_literals: Vec<TermId> = Vec::new();

        for &(literal, value) in &self.asserted {
            if !value {
                continue;
            }

            let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) else {
                continue;
            };

            if name != "=" || args.len() != 2 {
                continue;
            }

            let (var_coeffs, constant) =
                self.parse_equality_for_dioph(args[0], args[1], &term_to_idx);

            if var_coeffs.is_empty() {
                continue;
            }

            // Convert to rational row
            let mut row: Vec<BigRational> = vec![BigRational::zero(); num_vars];
            for (idx, coeff) in var_coeffs {
                row[idx] = BigRational::from(coeff);
            }
            matrix.push((row, BigRational::from(constant)));
            equality_literals.push(literal);
        }

        if matrix.is_empty() {
            return None;
        }

        if debug {
            eprintln!(
                "[ENUM] Built matrix with {} rows, {} cols",
                matrix.len(),
                num_vars
            );
        }

        // Gaussian elimination over rationals
        let mut pivot_cols: Vec<usize> = Vec::new();
        let mut pivot_row = 0;

        for col in 0..num_vars {
            // Find pivot in this column
            let mut best_row = None;
            for (row, row_data) in matrix.iter().enumerate().skip(pivot_row) {
                if !row_data.0[col].is_zero() {
                    best_row = Some(row);
                    break;
                }
            }

            let Some(prow) = best_row else {
                continue; // No pivot in this column, it's a free variable
            };

            // Swap rows
            if prow != pivot_row {
                matrix.swap(prow, pivot_row);
            }

            // Scale pivot row to make pivot = 1
            let pivot_val = matrix[pivot_row].0[col].clone();
            for c in 0..num_vars {
                matrix[pivot_row].0[c] = &matrix[pivot_row].0[c] / &pivot_val;
            }
            matrix[pivot_row].1 = &matrix[pivot_row].1 / &pivot_val;

            // Eliminate other rows
            for row in 0..matrix.len() {
                if row == pivot_row {
                    continue;
                }
                let factor = matrix[row].0[col].clone();
                if factor.is_zero() {
                    continue;
                }
                for c in 0..num_vars {
                    let sub = &factor * &matrix[pivot_row].0[c];
                    matrix[row].0[c] = &matrix[row].0[c] - &sub;
                }
                let sub = &factor * &matrix[pivot_row].1;
                matrix[row].1 = &matrix[row].1 - &sub;
            }

            pivot_cols.push(col);
            pivot_row += 1;

            if pivot_row >= matrix.len() {
                break;
            }
        }

        // Check for inconsistency: row of all zeros with non-zero constant
        for (row, constant) in &matrix {
            if row.iter().all(|c| c.is_zero()) && !constant.is_zero() {
                if debug {
                    eprintln!("[ENUM] Infeasible: 0 = {}", constant);
                }
                let conflict: Vec<TheoryLit> = equality_literals
                    .iter()
                    .map(|&lit| TheoryLit::new(lit, true))
                    .collect();
                return Some(conflict);
            }
        }

        // Identify free variables (columns without pivots)
        let pivot_set: HashSet<usize> = pivot_cols.iter().copied().collect();
        let free_vars: Vec<usize> = (0..num_vars).filter(|c| !pivot_set.contains(c)).collect();

        if debug {
            eprintln!(
                "[ENUM] Pivot cols: {:?}, free vars: {:?}",
                pivot_cols, free_vars
            );
        }

        if free_vars.is_empty() {
            // Fully determined - first build solution, then check inequalities
            if debug {
                eprintln!("[ENUM] System fully determined, checking solution");
            }

            // Build solution array
            let mut solution: Vec<(usize, BigInt)> = Vec::new();
            for (row_idx, &pivot_col) in pivot_cols.iter().enumerate() {
                let val = &matrix[row_idx].1;
                // Check if integer
                if !val.denom().is_one() {
                    if debug {
                        eprintln!(
                            "[ENUM] Non-integer solution for pivot col {}: {}",
                            pivot_col, val
                        );
                    }
                    let conflict: Vec<TheoryLit> = equality_literals
                        .iter()
                        .map(|&lit| TheoryLit::new(lit, true))
                        .collect();
                    return Some(conflict);
                }
                solution.push((pivot_col, val.numer().clone()));
            }

            // Check if solution satisfies all inequalities
            if self.check_solution_satisfies_inequalities(&solution, &idx_to_term) {
                if debug {
                    eprintln!("[ENUM] Solution satisfies inequalities, setting values");
                }
                // Set all bounds
                for (col, val) in solution {
                    let term_id = idx_to_term[col];
                    self.add_integer_bound(term_id, &val);
                }
                return None;
            } else {
                // Solution determined by equalities doesn't satisfy inequalities
                if debug {
                    eprintln!("[ENUM] Solution violates inequalities - infeasible");
                }
                let conflict: Vec<TheoryLit> = equality_literals
                    .iter()
                    .map(|&lit| TheoryLit::new(lit, true))
                    .collect();
                return Some(conflict);
            }
        }

        if free_vars.len() > 2 {
            // Too many free variables for enumeration
            return None;
        }

        // Build substitution expressions: pivot_var = constant - Σ(coeff * free_var)
        // After RREF, each pivot row has: x_pivot + Σ(coeff_f * x_free) = constant
        // So: x_pivot = constant - Σ(coeff_f * x_free)
        let mut substitutions: Vec<(usize, Vec<(usize, BigRational)>, BigRational)> = Vec::new();
        for (row_idx, &pivot_col) in pivot_cols.iter().enumerate() {
            let mut coeffs: Vec<(usize, BigRational)> = Vec::new();
            for &free_col in &free_vars {
                let c = &matrix[row_idx].0[free_col];
                if !c.is_zero() {
                    coeffs.push((free_col, -c.clone())); // Note: negated because pivot = const - Σ
                }
            }
            substitutions.push((pivot_col, coeffs, matrix[row_idx].1.clone()));
        }

        if debug {
            for (pivot, coeffs, constant) in &substitutions {
                eprintln!("[ENUM] x{} = {} + Σ{:?}", pivot, constant, coeffs);
            }
        }

        // Get bounds on free variables from inequalities
        // For each inequality, substitute pivot vars to get constraint on free vars
        let (free_lower, free_upper) =
            self.compute_free_var_bounds(&free_vars, &substitutions, &idx_to_term, &term_to_idx);

        if debug {
            for (i, &fv) in free_vars.iter().enumerate() {
                eprintln!(
                    "[ENUM] Free var x{}: bounds [{:?}, {:?}]",
                    fv,
                    free_lower.get(i),
                    free_upper.get(i)
                );
            }
        }

        // Check if bounds are finite and small enough for enumeration
        let max_enum_range: i64 = 10000;
        let max_total_points: i64 = 2_000_000;

        if free_vars.len() == 1 {
            let lo = free_lower.first().cloned().flatten();
            let hi = free_upper.first().cloned().flatten();

            if let (Some(lo_val), Some(hi_val)) = (&lo, &hi) {
                if lo_val > hi_val {
                    if debug {
                        eprintln!("[ENUM] Infeasible bounds: {} > {}", lo_val, hi_val);
                    }
                    let conflict: Vec<TheoryLit> = equality_literals
                        .iter()
                        .map(|&lit| TheoryLit::new(lit, true))
                        .collect();
                    return Some(conflict);
                }

                let range = hi_val - lo_val;
                if range <= BigInt::from(max_enum_range) {
                    // Enumerate!
                    if debug {
                        eprintln!("[ENUM] Enumerating free var in [{}, {}]", lo_val, hi_val);
                    }

                    let free_col = free_vars[0];
                    let mut t = lo_val.clone();
                    while &t <= hi_val {
                        // Compute all pivot vars
                        let t_rat = BigRational::from(t.clone());
                        let mut all_integer = true;
                        let mut solution: Vec<(usize, BigInt)> = Vec::new();

                        // Set free var
                        solution.push((free_col, t.clone()));

                        // Compute pivot vars
                        for (pivot_col, coeffs, constant) in &substitutions {
                            let mut val = constant.clone();
                            for (fc, coeff) in coeffs {
                                if *fc == free_col {
                                    val = &val + coeff * &t_rat;
                                }
                            }

                            if !val.denom().is_one() {
                                all_integer = false;
                                break;
                            }
                            solution.push((*pivot_col, val.numer().clone()));
                        }

                        if all_integer {
                            // Check inequalities
                            if self.check_solution_satisfies_inequalities(&solution, &idx_to_term) {
                                if debug {
                                    eprintln!("[ENUM] Found valid solution: {:?}", solution);
                                }
                                // Set all bounds
                                for (col, val) in solution {
                                    let term_id = idx_to_term[col];
                                    self.add_integer_bound(term_id, &val);
                                }
                                return None;
                            }
                        }

                        t += BigInt::one();
                    }

                    // No solution found
                    if debug {
                        eprintln!("[ENUM] No valid solution found in range");
                    }
                    let conflict: Vec<TheoryLit> = equality_literals
                        .iter()
                        .map(|&lit| TheoryLit::new(lit, true))
                        .collect();
                    return Some(conflict);
                }
            }
        }

        if free_vars.len() == 2 {
            let lo0 = free_lower.first().cloned().flatten();
            let hi0 = free_upper.first().cloned().flatten();
            let lo1 = free_lower.get(1).cloned().flatten();
            let hi1 = free_upper.get(1).cloned().flatten();

            if let (Some(lo0), Some(hi0), Some(lo1), Some(hi1)) = (&lo0, &hi0, &lo1, &hi1) {
                if lo0 > hi0 || lo1 > hi1 {
                    if debug {
                        eprintln!(
                            "[ENUM] Infeasible bounds: [{}, {}] x [{}, {}]",
                            lo0, hi0, lo1, hi1
                        );
                    }
                    let conflict: Vec<TheoryLit> = equality_literals
                        .iter()
                        .map(|&lit| TheoryLit::new(lit, true))
                        .collect();
                    return Some(conflict);
                }

                let range0 = hi0 - lo0;
                let range1 = hi1 - lo1;
                if range0 <= BigInt::from(max_enum_range) && range1 <= BigInt::from(max_enum_range)
                {
                    let count0 = range0 + BigInt::one();
                    let count1 = range1 + BigInt::one();
                    let total_points = &count0 * &count1;

                    if total_points <= BigInt::from(max_total_points) {
                        if debug {
                            eprintln!(
                                "[ENUM] Enumerating 2D grid: x{} in [{}, {}], x{} in [{}, {}] ({} points)",
                                free_vars[0],
                                lo0,
                                hi0,
                                free_vars[1],
                                lo1,
                                hi1,
                                total_points
                            );
                        }

                        let free_col0 = free_vars[0];
                        let free_col1 = free_vars[1];

                        let mut t0 = lo0.clone();
                        while &t0 <= hi0 {
                            let t0_rat = BigRational::from(t0.clone());
                            let mut t1 = lo1.clone();
                            while &t1 <= hi1 {
                                let t1_rat = BigRational::from(t1.clone());

                                // Compute all pivot vars
                                let mut all_integer = true;
                                let mut solution: Vec<(usize, BigInt)> = Vec::new();

                                // Set free vars
                                solution.push((free_col0, t0.clone()));
                                solution.push((free_col1, t1.clone()));

                                // Compute pivot vars
                                for (pivot_col, coeffs, constant) in &substitutions {
                                    let mut val = constant.clone();
                                    for (fc, coeff) in coeffs {
                                        if *fc == free_col0 {
                                            val = &val + coeff * &t0_rat;
                                        } else if *fc == free_col1 {
                                            val = &val + coeff * &t1_rat;
                                        }
                                    }

                                    if !val.denom().is_one() {
                                        all_integer = false;
                                        break;
                                    }
                                    solution.push((*pivot_col, val.numer().clone()));
                                }

                                if all_integer {
                                    // Check inequalities
                                    if self.check_solution_satisfies_inequalities(
                                        &solution,
                                        &idx_to_term,
                                    ) {
                                        if debug {
                                            eprintln!(
                                                "[ENUM] Found valid solution: {:?}",
                                                solution
                                            );
                                        }
                                        // Set all bounds
                                        for (col, val) in solution {
                                            let term_id = idx_to_term[col];
                                            self.add_integer_bound(term_id, &val);
                                        }
                                        return None;
                                    }
                                }

                                t1 += BigInt::one();
                            }
                            t0 += BigInt::one();
                        }

                        // No solution found
                        if debug {
                            eprintln!("[ENUM] No valid solution found in 2D range");
                        }
                        let conflict: Vec<TheoryLit> = equality_literals
                            .iter()
                            .map(|&lit| TheoryLit::new(lit, true))
                            .collect();
                        return Some(conflict);
                    }
                }
            }
        }

        // Can't cheaply enumerate (or no finite bounds found).
        None
    }

    /// Compute bounds on free variables by substituting into inequalities.
    fn compute_free_var_bounds(
        &self,
        free_vars: &[usize],
        substitutions: &[(usize, Vec<(usize, BigRational)>, BigRational)],
        idx_to_term: &[TermId],
        term_to_idx: &HashMap<TermId, usize>,
    ) -> (Vec<Option<BigInt>>, Vec<Option<BigInt>>) {
        let debug = std::env::var("Z4_DEBUG_ENUM").is_ok();
        let mut lower: Vec<Option<BigInt>> = vec![None; free_vars.len()];
        let mut upper: Vec<Option<BigInt>> = vec![None; free_vars.len()];

        // Get direct bounds from LRA on free variables
        for (i, &free_col) in free_vars.iter().enumerate() {
            if free_col < idx_to_term.len() {
                let term_id = idx_to_term[free_col];
                if let Some((lb, ub)) = self.lra.get_bounds(term_id) {
                    if let Some(lb) = lb {
                        let lb_int = Self::ceil_bigint(&lb.value);
                        lower[i] = Some(lb_int);
                    }
                    if let Some(ub) = ub {
                        let ub_int = Self::floor_bigint(&ub.value);
                        upper[i] = Some(ub_int);
                    }
                }
            }
        }

        // Also derive bounds from pivot variable bounds
        // If x_pivot = constant + coeff * t, and x_pivot has bounds, we can derive t bounds
        for (pivot_col, coeffs, constant) in substitutions {
            if *pivot_col >= idx_to_term.len() {
                continue;
            }
            let pivot_term = idx_to_term[*pivot_col];

            // Get pivot bounds from LRA
            let (pivot_lb, pivot_ub) = match self.lra.get_bounds(pivot_term) {
                Some((lb, ub)) => (lb.map(|b| b.value), ub.map(|b| b.value)),
                None => (None, None),
            };

            // For single free variable case with x_pivot = constant + coeff * t
            if coeffs.len() == 1 && free_vars.len() == 1 {
                let (fc, coeff) = &coeffs[0];
                let i = free_vars.iter().position(|&fv| fv == *fc).unwrap_or(0);

                // x_pivot >= lb => constant + coeff * t >= lb => t >= (lb - constant) / coeff (if coeff > 0)
                if coeff.is_positive() {
                    if let Some(ref lb) = pivot_lb {
                        // t >= ceil((lb - constant) / coeff)
                        let bound = (lb - constant) / coeff;
                        let bound_int = Self::ceil_bigint(&bound);
                        if debug {
                            eprintln!(
                                "[ENUM] From pivot x{} >= {}: t >= {}",
                                pivot_col, lb, bound_int
                            );
                        }
                        lower[i] = Some(match &lower[i] {
                            Some(l) => l.clone().max(bound_int),
                            None => bound_int,
                        });
                    }
                    if let Some(ref ub) = pivot_ub {
                        // t <= floor((ub - constant) / coeff)
                        let bound = (ub - constant) / coeff;
                        let bound_int = Self::floor_bigint(&bound);
                        if debug {
                            eprintln!(
                                "[ENUM] From pivot x{} <= {}: t <= {}",
                                pivot_col, ub, bound_int
                            );
                        }
                        upper[i] = Some(match &upper[i] {
                            Some(u) => u.clone().min(bound_int),
                            None => bound_int,
                        });
                    }
                } else if coeff.is_negative() {
                    // t <= (lb - constant) / coeff (direction flips)
                    if let Some(ref lb) = pivot_lb {
                        let bound = (lb - constant) / coeff;
                        let bound_int = Self::floor_bigint(&bound);
                        if debug {
                            eprintln!(
                                "[ENUM] From pivot x{} >= {} (neg coeff): t <= {}",
                                pivot_col, lb, bound_int
                            );
                        }
                        upper[i] = Some(match &upper[i] {
                            Some(u) => u.clone().min(bound_int),
                            None => bound_int,
                        });
                    }
                    if let Some(ref ub) = pivot_ub {
                        let bound = (ub - constant) / coeff;
                        let bound_int = Self::ceil_bigint(&bound);
                        if debug {
                            eprintln!(
                                "[ENUM] From pivot x{} <= {} (neg coeff): t >= {}",
                                pivot_col, ub, bound_int
                            );
                        }
                        lower[i] = Some(match &lower[i] {
                            Some(l) => l.clone().max(bound_int),
                            None => bound_int,
                        });
                    }
                }
            }
        }

        // Derive bounds by substituting into original inequalities
        // For single free variable, each inequality becomes: coeff * t <= bound or coeff * t >= bound
        if free_vars.len() == 1 {
            let free_col = free_vars[0];

            for &(literal, value) in &self.asserted {
                if !value {
                    continue;
                }

                let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) else {
                    continue;
                };

                if args.len() != 2 {
                    continue;
                }

                // Skip equalities (already handled by Gaussian elimination)
                if name == "=" {
                    continue;
                }

                // Parse the inequality as: lhs - rhs OP 0
                // After substitution: (constant_part + free_coeff * t) OP 0
                let (constant_part, free_coeff) = self.substitute_and_simplify(
                    args[0],
                    args[1],
                    substitutions,
                    free_col,
                    term_to_idx,
                    idx_to_term,
                );

                if free_coeff.is_zero() {
                    continue; // Inequality doesn't involve the free variable
                }

                // Derive bound based on operator and coefficient sign
                // lhs - rhs >= 0 means: constant_part + free_coeff * t >= 0
                // lhs - rhs <= 0 means: constant_part + free_coeff * t <= 0
                match name.as_str() {
                    ">=" => {
                        // constant_part + free_coeff * t >= 0
                        // free_coeff * t >= -constant_part
                        if free_coeff.is_positive() {
                            // t >= -constant_part / free_coeff
                            let bound = -&constant_part / &free_coeff;
                            let bound_int = Self::ceil_bigint(&bound);
                            if debug {
                                eprintln!("[ENUM] From >= inequality: t >= {}", bound_int);
                            }
                            lower[0] = Some(match &lower[0] {
                                Some(l) => l.clone().max(bound_int),
                                None => bound_int,
                            });
                        } else {
                            // t <= -constant_part / free_coeff (flipped)
                            let bound = -&constant_part / &free_coeff;
                            let bound_int = Self::floor_bigint(&bound);
                            if debug {
                                eprintln!("[ENUM] From >= inequality (neg): t <= {}", bound_int);
                            }
                            upper[0] = Some(match &upper[0] {
                                Some(u) => u.clone().min(bound_int),
                                None => bound_int,
                            });
                        }
                    }
                    "<=" => {
                        // constant_part + free_coeff * t <= 0
                        // free_coeff * t <= -constant_part
                        if free_coeff.is_positive() {
                            // t <= -constant_part / free_coeff
                            let bound = -&constant_part / &free_coeff;
                            let bound_int = Self::floor_bigint(&bound);
                            if debug {
                                eprintln!("[ENUM] From <= inequality: t <= {}", bound_int);
                            }
                            upper[0] = Some(match &upper[0] {
                                Some(u) => u.clone().min(bound_int),
                                None => bound_int,
                            });
                        } else {
                            // t >= -constant_part / free_coeff (flipped)
                            let bound = -&constant_part / &free_coeff;
                            let bound_int = Self::ceil_bigint(&bound);
                            if debug {
                                eprintln!("[ENUM] From <= inequality (neg): t >= {}", bound_int);
                            }
                            lower[0] = Some(match &lower[0] {
                                Some(l) => l.clone().max(bound_int),
                                None => bound_int,
                            });
                        }
                    }
                    ">" => {
                        // constant_part + free_coeff * t > 0 => same as >= but with +1/-1 adjustment
                        if free_coeff.is_positive() {
                            let bound = -&constant_part / &free_coeff;
                            let bound_int = Self::ceil_bigint(&bound);
                            // Strict inequality: if bound is exactly an integer, add 1
                            let adjusted = if Self::is_integer(&(-&constant_part / &free_coeff)) {
                                &bound_int + BigInt::one()
                            } else {
                                bound_int
                            };
                            if debug {
                                eprintln!("[ENUM] From > inequality: t >= {}", adjusted);
                            }
                            lower[0] = Some(match &lower[0] {
                                Some(l) => l.clone().max(adjusted),
                                None => adjusted,
                            });
                        } else {
                            let bound = -&constant_part / &free_coeff;
                            let bound_int = Self::floor_bigint(&bound);
                            let adjusted = if Self::is_integer(&(-&constant_part / &free_coeff)) {
                                &bound_int - BigInt::one()
                            } else {
                                bound_int
                            };
                            if debug {
                                eprintln!("[ENUM] From > inequality (neg): t <= {}", adjusted);
                            }
                            upper[0] = Some(match &upper[0] {
                                Some(u) => u.clone().min(adjusted),
                                None => adjusted,
                            });
                        }
                    }
                    "<" => {
                        if free_coeff.is_positive() {
                            let bound = -&constant_part / &free_coeff;
                            let bound_int = Self::floor_bigint(&bound);
                            let adjusted = if Self::is_integer(&(-&constant_part / &free_coeff)) {
                                &bound_int - BigInt::one()
                            } else {
                                bound_int
                            };
                            if debug {
                                eprintln!("[ENUM] From < inequality: t <= {}", adjusted);
                            }
                            upper[0] = Some(match &upper[0] {
                                Some(u) => u.clone().min(adjusted),
                                None => adjusted,
                            });
                        } else {
                            let bound = -&constant_part / &free_coeff;
                            let bound_int = Self::ceil_bigint(&bound);
                            let adjusted = if Self::is_integer(&(-&constant_part / &free_coeff)) {
                                &bound_int + BigInt::one()
                            } else {
                                bound_int
                            };
                            if debug {
                                eprintln!("[ENUM] From < inequality (neg): t >= {}", adjusted);
                            }
                            lower[0] = Some(match &lower[0] {
                                Some(l) => l.clone().max(adjusted),
                                None => adjusted,
                            });
                        }
                    }
                    _ => {}
                }
            }
        } else if free_vars.len() == 2 {
            #[derive(Clone, Debug)]
            struct Constraint {
                op: IneqOp,
                constant: BigRational,
                c0: BigRational,
                c1: BigRational,
            }

            let fv0 = free_vars[0];
            let fv1 = free_vars[1];

            let mut constraints: Vec<Constraint> = Vec::new();

            // Constraints from original asserted inequalities after substitution.
            for &(literal, value) in &self.asserted {
                if !value {
                    continue;
                }

                let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) else {
                    continue;
                };

                if args.len() != 2 {
                    continue;
                }

                // Skip equalities (already handled by Gaussian elimination)
                if name == "=" {
                    continue;
                }

                let op = match name.as_str() {
                    ">=" => IneqOp::Ge,
                    "<=" => IneqOp::Le,
                    ">" => IneqOp::Gt,
                    "<" => IneqOp::Lt,
                    _ => continue,
                };

                let (constant_part, coeffs) = self.substitute_and_simplify_multi(
                    args[0],
                    args[1],
                    substitutions,
                    free_vars,
                    term_to_idx,
                );

                if coeffs[0].is_zero() && coeffs[1].is_zero() {
                    continue;
                }

                constraints.push(Constraint {
                    op,
                    constant: constant_part,
                    c0: coeffs[0].clone(),
                    c1: coeffs[1].clone(),
                });
            }

            // Constraints from pivot-variable bounds (x_pivot = constant + c0*t0 + c1*t1).
            for (pivot_col, pivot_coeffs, pivot_constant) in substitutions {
                if *pivot_col >= idx_to_term.len() {
                    continue;
                }
                let pivot_term = idx_to_term[*pivot_col];

                let (pivot_lb, pivot_ub) = match self.lra.get_bounds(pivot_term) {
                    Some((lb, ub)) => (lb.map(|b| b.value), ub.map(|b| b.value)),
                    None => (None, None),
                };

                let mut c0 = BigRational::zero();
                let mut c1 = BigRational::zero();
                for (fc, c) in pivot_coeffs {
                    if *fc == fv0 {
                        c0 = c.clone();
                    } else if *fc == fv1 {
                        c1 = c.clone();
                    }
                }
                if c0.is_zero() && c1.is_zero() {
                    continue;
                }

                if let Some(lb) = pivot_lb {
                    // pivot >= lb  =>  (pivot_constant - lb) + c0*t0 + c1*t1 >= 0
                    constraints.push(Constraint {
                        op: IneqOp::Ge,
                        constant: pivot_constant - &lb,
                        c0: c0.clone(),
                        c1: c1.clone(),
                    });
                }

                if let Some(ub) = pivot_ub {
                    // pivot <= ub  =>  (pivot_constant - ub) + c0*t0 + c1*t1 <= 0
                    constraints.push(Constraint {
                        op: IneqOp::Le,
                        constant: pivot_constant - &ub,
                        c0,
                        c1,
                    });
                }
            }

            let mut changed = true;
            let mut iterations = 0usize;

            while changed && iterations < 20 {
                changed = false;
                iterations += 1;

                for c in &constraints {
                    // Update t0 using t1 bounds.
                    let other_l = lower[1].clone();
                    let other_u = upper[1].clone();
                    changed |= Self::tighten_two_var_bound(
                        c.op,
                        &c.constant,
                        &c.c0,
                        &c.c1,
                        &other_l,
                        &other_u,
                        &mut lower[0],
                        &mut upper[0],
                    );

                    // Update t1 using t0 bounds.
                    let other_l = lower[0].clone();
                    let other_u = upper[0].clone();
                    changed |= Self::tighten_two_var_bound(
                        c.op,
                        &c.constant,
                        &c.c1,
                        &c.c0,
                        &other_l,
                        &other_u,
                        &mut lower[1],
                        &mut upper[1],
                    );
                }
            }

            if debug {
                eprintln!(
                    "[ENUM] 2D bound propagation iterations: {}, bounds t0=[{:?},{:?}] t1=[{:?},{:?}] ({} constraints)",
                    iterations,
                    lower[0],
                    upper[0],
                    lower[1],
                    upper[1],
                    constraints.len()
                );
            }
        }

        (lower, upper)
    }

    fn tighten_two_var_bound(
        op: IneqOp,
        constant: &BigRational,
        a: &BigRational,
        b: &BigRational,
        other_lower: &Option<BigInt>,
        other_upper: &Option<BigInt>,
        this_lower: &mut Option<BigInt>,
        this_upper: &mut Option<BigInt>,
    ) -> bool {
        #[derive(Clone, Copy, Debug)]
        enum BoundKind {
            LowerInclusive,
            LowerStrict,
            UpperInclusive,
            UpperStrict,
        }

        let (bound_kind, use_min_over_other) = match op {
            IneqOp::Ge => {
                if a.is_positive() {
                    (BoundKind::LowerInclusive, true)
                } else if a.is_negative() {
                    (BoundKind::UpperInclusive, false)
                } else {
                    return false;
                }
            }
            IneqOp::Gt => {
                if a.is_positive() {
                    (BoundKind::LowerStrict, true)
                } else if a.is_negative() {
                    (BoundKind::UpperStrict, false)
                } else {
                    return false;
                }
            }
            IneqOp::Le => {
                if a.is_positive() {
                    (BoundKind::UpperInclusive, false)
                } else if a.is_negative() {
                    (BoundKind::LowerInclusive, true)
                } else {
                    return false;
                }
            }
            IneqOp::Lt => {
                if a.is_positive() {
                    (BoundKind::UpperStrict, false)
                } else if a.is_negative() {
                    (BoundKind::LowerStrict, true)
                } else {
                    return false;
                }
            }
        };

        // rhs(other) = -constant - b*other, so a*this OP rhs(other).
        let (rhs_min, rhs_max) = if b.is_zero() {
            let rhs = -constant.clone();
            (rhs.clone(), rhs)
        } else {
            let (Some(ol), Some(ou)) = (other_lower.as_ref(), other_upper.as_ref()) else {
                return false;
            };

            let ol = BigRational::from(ol.clone());
            let ou = BigRational::from(ou.clone());

            let rhs_l = -constant - b * &ol;
            let rhs_u = -constant - b * &ou;
            if rhs_l <= rhs_u {
                (rhs_l, rhs_u)
            } else {
                (rhs_u, rhs_l)
            }
        };

        // Pick min/max of rhs/a over the other-var interval (account for a's sign).
        let min_over_other = if a.is_positive() {
            &rhs_min / a
        } else {
            &rhs_max / a
        };
        let max_over_other = if a.is_positive() {
            &rhs_max / a
        } else {
            &rhs_min / a
        };

        let bound = if use_min_over_other {
            min_over_other
        } else {
            max_over_other
        };

        let mut changed = false;
        match bound_kind {
            BoundKind::LowerInclusive => {
                let new_lb = Self::ceil_bigint(&bound);
                let next = match this_lower.as_ref() {
                    Some(cur) => cur.clone().max(new_lb),
                    None => new_lb,
                };
                if this_lower.as_ref() != Some(&next) {
                    *this_lower = Some(next);
                    changed = true;
                }
            }
            BoundKind::LowerStrict => {
                let mut new_lb = Self::ceil_bigint(&bound);
                if Self::is_integer(&bound) {
                    new_lb += BigInt::one();
                }
                let next = match this_lower.as_ref() {
                    Some(cur) => cur.clone().max(new_lb),
                    None => new_lb,
                };
                if this_lower.as_ref() != Some(&next) {
                    *this_lower = Some(next);
                    changed = true;
                }
            }
            BoundKind::UpperInclusive => {
                let new_ub = Self::floor_bigint(&bound);
                let next = match this_upper.as_ref() {
                    Some(cur) => cur.clone().min(new_ub),
                    None => new_ub,
                };
                if this_upper.as_ref() != Some(&next) {
                    *this_upper = Some(next);
                    changed = true;
                }
            }
            BoundKind::UpperStrict => {
                let mut new_ub = Self::floor_bigint(&bound);
                if Self::is_integer(&bound) {
                    new_ub -= BigInt::one();
                }
                let next = match this_upper.as_ref() {
                    Some(cur) => cur.clone().min(new_ub),
                    None => new_ub,
                };
                if this_upper.as_ref() != Some(&next) {
                    *this_upper = Some(next);
                    changed = true;
                }
            }
        }

        changed
    }

    fn substitute_and_simplify_multi(
        &self,
        lhs: TermId,
        rhs: TermId,
        substitutions: &[(usize, Vec<(usize, BigRational)>, BigRational)],
        free_cols: &[usize],
        term_to_idx: &HashMap<TermId, usize>,
    ) -> (BigRational, [BigRational; 2]) {
        debug_assert!(free_cols.len() == 2);
        let free_col0 = free_cols[0];
        let free_col1 = free_cols[1];

        // Parse lhs - rhs into coefficients
        let mut var_coeffs: HashMap<usize, BigRational> = HashMap::new();
        let mut constant = BigRational::zero();

        self.collect_rational_terms(
            lhs,
            &BigRational::one(),
            &mut var_coeffs,
            &mut constant,
            term_to_idx,
        );
        self.collect_rational_terms(
            rhs,
            &-BigRational::one(),
            &mut var_coeffs,
            &mut constant,
            term_to_idx,
        );

        let mut final_constant = constant;
        let mut free_coeffs = [BigRational::zero(), BigRational::zero()];

        free_coeffs[0] = var_coeffs.remove(&free_col0).unwrap_or_default();
        free_coeffs[1] = var_coeffs.remove(&free_col1).unwrap_or_default();

        // Substitute pivot variables
        for (pivot_col, pivot_coeffs, pivot_constant) in substitutions {
            if let Some(pivot_coeff) = var_coeffs.remove(pivot_col) {
                final_constant = &final_constant + &pivot_coeff * pivot_constant;
                for (fc, c) in pivot_coeffs {
                    if *fc == free_col0 {
                        free_coeffs[0] = &free_coeffs[0] + &pivot_coeff * c;
                    } else if *fc == free_col1 {
                        free_coeffs[1] = &free_coeffs[1] + &pivot_coeff * c;
                    }
                }
            }
        }

        (final_constant, free_coeffs)
    }

    /// Substitute pivot variable expressions into an inequality and simplify.
    /// Returns (constant_part, free_var_coeff) such that:
    /// lhs - rhs = constant_part + free_var_coeff * free_var
    fn substitute_and_simplify(
        &self,
        lhs: TermId,
        rhs: TermId,
        substitutions: &[(usize, Vec<(usize, BigRational)>, BigRational)],
        free_col: usize,
        term_to_idx: &HashMap<TermId, usize>,
        _idx_to_term: &[TermId],
    ) -> (BigRational, BigRational) {
        // Parse lhs - rhs into coefficients
        let mut var_coeffs: HashMap<usize, BigRational> = HashMap::new();
        let mut constant = BigRational::zero();

        self.collect_rational_terms(
            lhs,
            &BigRational::one(),
            &mut var_coeffs,
            &mut constant,
            term_to_idx,
        );
        self.collect_rational_terms(
            rhs,
            &-BigRational::one(),
            &mut var_coeffs,
            &mut constant,
            term_to_idx,
        );

        // Now substitute pivot variables
        // For each pivot var p: if var_coeffs[p] != 0, replace with substitution expression
        let mut final_constant = constant;
        let mut free_coeff = var_coeffs.remove(&free_col).unwrap_or_default();

        for (pivot_col, pivot_coeffs, pivot_constant) in substitutions {
            if let Some(pivot_coeff) = var_coeffs.remove(pivot_col) {
                // x_pivot = pivot_constant + Σ(coeff * free_var)
                // Replace pivot_coeff * x_pivot with:
                // pivot_coeff * pivot_constant + pivot_coeff * Σ(coeff * free_var)
                final_constant = &final_constant + &pivot_coeff * pivot_constant;

                for (fc, c) in pivot_coeffs {
                    if *fc == free_col {
                        free_coeff = &free_coeff + &pivot_coeff * c;
                    }
                }
            }
        }

        (final_constant, free_coeff)
    }

    /// Collect rational coefficients from a term expression.
    fn collect_rational_terms(
        &self,
        term: TermId,
        scale: &BigRational,
        coeffs: &mut HashMap<usize, BigRational>,
        constant: &mut BigRational,
        term_to_idx: &HashMap<TermId, usize>,
    ) {
        match self.terms.get(term) {
            TermData::Const(Constant::Int(n)) => {
                *constant = &*constant + scale * BigRational::from(n.clone());
            }
            TermData::Const(Constant::Rational(r)) => {
                *constant = &*constant + scale * &r.0;
            }
            TermData::Var(_, _) => {
                if let Some(&idx) = term_to_idx.get(&term) {
                    *coeffs.entry(idx).or_insert_with(BigRational::zero) += scale;
                }
            }
            TermData::App(Symbol::Named(name), args) => {
                match name.as_str() {
                    "+" => {
                        for &arg in args {
                            self.collect_rational_terms(arg, scale, coeffs, constant, term_to_idx);
                        }
                    }
                    "-" if args.len() == 1 => {
                        self.collect_rational_terms(
                            args[0],
                            &-scale.clone(),
                            coeffs,
                            constant,
                            term_to_idx,
                        );
                    }
                    "-" if args.len() >= 2 => {
                        self.collect_rational_terms(args[0], scale, coeffs, constant, term_to_idx);
                        for &arg in &args[1..] {
                            self.collect_rational_terms(
                                arg,
                                &-scale.clone(),
                                coeffs,
                                constant,
                                term_to_idx,
                            );
                        }
                    }
                    "*" => {
                        let mut const_factor = BigRational::one();
                        let mut var_args = Vec::new();

                        for &arg in args {
                            if let Some(c) = self.extract_rational_constant(arg) {
                                const_factor *= c;
                            } else {
                                var_args.push(arg);
                            }
                        }

                        let new_scale = scale * &const_factor;

                        if var_args.is_empty() {
                            *constant = &*constant + &new_scale;
                        } else if var_args.len() == 1 {
                            self.collect_rational_terms(
                                var_args[0],
                                &new_scale,
                                coeffs,
                                constant,
                                term_to_idx,
                            );
                        }
                        // Non-linear terms are ignored
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Check if a candidate solution satisfies all asserted inequalities.
    fn check_solution_satisfies_inequalities(
        &self,
        solution: &[(usize, BigInt)],
        idx_to_term: &[TermId],
    ) -> bool {
        let debug = std::env::var("Z4_DEBUG_ENUM").is_ok();

        // Build a map from term to value
        let mut values: HashMap<TermId, BigInt> = HashMap::new();
        for &(col, ref val) in solution {
            if col < idx_to_term.len() {
                values.insert(idx_to_term[col], val.clone());
            }
        }

        // Check all asserted inequalities
        for &(literal, value) in &self.asserted {
            if !value {
                continue;
            }

            let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) else {
                continue;
            };

            if args.len() != 2 {
                continue;
            }

            // Evaluate lhs and rhs
            let lhs_val = self.evaluate_linear_expr(&values, args[0]);
            let rhs_val = self.evaluate_linear_expr(&values, args[1]);

            let (Some(lhs), Some(rhs)) = (lhs_val, rhs_val) else {
                continue; // Can't evaluate, skip
            };

            let satisfied = match name.as_str() {
                ">=" => lhs >= rhs,
                "<=" => lhs <= rhs,
                ">" => lhs > rhs,
                "<" => lhs < rhs,
                "=" => lhs == rhs,
                _ => true, // Unknown operator, assume satisfied
            };

            if !satisfied {
                if debug {
                    eprintln!(
                        "[ENUM] Constraint {} {} {} not satisfied ({} vs {})",
                        args[0].0, name, args[1].0, lhs, rhs
                    );
                }
                return false;
            }
        }

        true
    }

    /// Evaluate a linear expression given variable assignments.
    fn evaluate_linear_expr(
        &self,
        values: &HashMap<TermId, BigInt>,
        term: TermId,
    ) -> Option<BigInt> {
        match self.terms.get(term) {
            TermData::Const(Constant::Int(n)) => Some(n.clone()),
            TermData::Const(Constant::Rational(r)) => {
                if r.0.denom().is_one() {
                    Some(r.0.numer().clone())
                } else {
                    None
                }
            }
            TermData::Var(_, _) => values.get(&term).cloned(),
            TermData::App(Symbol::Named(name), args) => match name.as_str() {
                "+" => {
                    let mut sum = BigInt::zero();
                    for &arg in args {
                        sum += self.evaluate_linear_expr(values, arg)?;
                    }
                    Some(sum)
                }
                "-" if args.len() == 1 => Some(-self.evaluate_linear_expr(values, args[0])?),
                "-" if args.len() >= 2 => {
                    let mut result = self.evaluate_linear_expr(values, args[0])?;
                    for &arg in &args[1..] {
                        result -= self.evaluate_linear_expr(values, arg)?;
                    }
                    Some(result)
                }
                "*" => {
                    let mut product = BigInt::one();
                    for &arg in args {
                        product *= self.evaluate_linear_expr(values, arg)?;
                    }
                    Some(product)
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Propagate bounds through cached substitutions.
    ///
    /// For each cached substitution `var = c + Σ(a_i * dep_i)`, uses current
    /// bounds on dep_i to derive tighter bounds on var.
    fn propagate_bounds_through_substitutions(&mut self) {
        let debug = std::env::var("Z4_DEBUG_DIOPH").is_ok();

        // Iterate over cloned substitutions to avoid borrow issues
        for (term_id, coeffs, constant) in self.dioph_cached_substitutions.clone() {
            // Compute implied bounds from the substitution
            // var = constant + Σ(coeff_i * dep_i)
            // lower(var) = constant + Σ(coeff_i > 0 ? coeff_i * lower(dep_i) : coeff_i * upper(dep_i))
            // upper(var) = constant + Σ(coeff_i > 0 ? coeff_i * upper(dep_i) : coeff_i * lower(dep_i))
            let mut implied_lower: Option<BigRational> = Some(BigRational::from(constant.clone()));
            let mut implied_upper: Option<BigRational> = Some(BigRational::from(constant.clone()));

            for (dep_term, coeff) in &coeffs {
                let (dep_lower, dep_upper) = match self.lra.get_bounds(*dep_term) {
                    Some((l, u)) => (l.map(|b| b.value), u.map(|b| b.value)),
                    None => (None, None),
                };

                let coeff_rat = BigRational::from(coeff.clone());

                if coeff.is_positive() {
                    if let (Some(ref mut il), Some(dl)) = (&mut implied_lower, dep_lower) {
                        *il += &coeff_rat * &dl;
                    } else {
                        implied_lower = None;
                    }
                    if let (Some(ref mut iu), Some(du)) = (&mut implied_upper, dep_upper) {
                        *iu += &coeff_rat * &du;
                    } else {
                        implied_upper = None;
                    }
                } else {
                    if let (Some(ref mut il), Some(du)) = (&mut implied_lower, dep_upper) {
                        *il += &coeff_rat * &du;
                    } else {
                        implied_lower = None;
                    }
                    if let (Some(ref mut iu), Some(dl)) = (&mut implied_upper, dep_lower) {
                        *iu += &coeff_rat * &dl;
                    } else {
                        implied_upper = None;
                    }
                }
            }

            // Convert to integer bounds and apply if tighter
            if let Some(il) = implied_lower {
                let il_int = Self::ceil_bigint(&il);
                if let Some((Some(current_lb), _)) = self.lra.get_bounds(term_id) {
                    let current_lb_int = Self::ceil_bigint(&current_lb.value);
                    if il_int > current_lb_int {
                        if debug {
                            eprintln!(
                                "[DIOPH] Propagating lower: {:?} {} -> {}",
                                term_id, current_lb_int, il_int
                            );
                        }
                        self.add_integer_bound(term_id, &il_int);
                    }
                }
            }

            if let Some(iu) = implied_upper {
                let iu_int = Self::floor_bigint(&iu);
                if let Some((_, Some(current_ub))) = self.lra.get_bounds(term_id) {
                    let current_ub_int = Self::floor_bigint(&current_ub.value);
                    if iu_int < current_ub_int {
                        if debug {
                            eprintln!(
                                "[DIOPH] Propagating upper: {:?} {} -> {}",
                                term_id, current_ub_int, iu_int
                            );
                        }
                        self.add_integer_bound(term_id, &iu_int);
                    }
                }
            }
        }
    }

    /// Check modular constraints derived from substitutions against bounds.
    ///
    /// For each substitution `var = c + Σ(a_i * x_i)`, if the GCD of coefficients
    /// is > 1, then `var ≡ c (mod GCD)`. Combined with bounds, this can detect
    /// infeasibility when no valid integer exists in the bounds.
    ///
    /// This is critical for CHC solving where mod constraints interact with bounds.
    fn check_modular_constraint_conflict(&self) -> Option<Vec<TheoryLit>> {
        let debug = std::env::var("Z4_DEBUG_MOD").is_ok();

        for (term_id, coeffs, constant) in &self.dioph_cached_substitutions {
            // Compute GCD of all coefficients in the substitution
            let mut gcd = BigInt::zero();
            for (_, coeff) in coeffs {
                if gcd.is_zero() {
                    gcd = coeff.abs();
                } else {
                    gcd = gcd.gcd(&coeff.abs());
                }
                // Early exit if GCD becomes 1
                if gcd.is_one() {
                    break;
                }
            }

            // If GCD > 1, we have a modular constraint: var ≡ constant (mod gcd)
            if gcd > BigInt::one() {
                // Compute residue: constant mod gcd, ensuring non-negative
                let residue = {
                    let r = constant % &gcd;
                    if r < BigInt::zero() {
                        r + &gcd
                    } else {
                        r
                    }
                };

                if debug {
                    eprintln!(
                        "[MOD] Variable {:?} ≡ {} (mod {}) from substitution",
                        term_id, residue, gcd
                    );
                }

                // Get current bounds for the variable
                if let Some((lb_opt, ub_opt)) = self.lra.get_bounds(*term_id) {
                    // For integers, effective bounds are ceil(lb) to floor(ub)
                    let effective_lb = lb_opt.map(|b| Self::ceil_bigint(&b.value));
                    let effective_ub = ub_opt.map(|b| Self::floor_bigint(&b.value));

                    if let (Some(lb), Some(ub)) = (&effective_lb, &effective_ub) {
                        if debug {
                            eprintln!("[MOD] Variable {:?} bounds: [{}, {}]", term_id, lb, ub);
                        }

                        // Find the first valid integer >= lb that satisfies var ≡ residue (mod gcd)
                        // first_valid = lb + ((residue - lb) mod gcd)
                        let diff = &residue - lb;
                        let adjustment = {
                            let r = &diff % &gcd;
                            if r < BigInt::zero() {
                                r + &gcd
                            } else {
                                r
                            }
                        };
                        let first_valid = lb + adjustment;

                        // Check if any valid integer exists in [lb, ub]
                        if &first_valid > ub {
                            if debug {
                                eprintln!(
                                    "[MOD] UNSAT: no integer in [{}, {}] satisfies ≡ {} (mod {})",
                                    lb, ub, residue, gcd
                                );
                            }
                            // Return conflict: all asserted literals
                            let conflict: Vec<TheoryLit> = self
                                .asserted
                                .iter()
                                .map(|&(lit, val)| TheoryLit::new(lit, val))
                                .collect();
                            return Some(conflict);
                        }

                        // If bounds are tight enough, we can derive exact value
                        if first_valid == *ub || &first_valid + &gcd > *ub {
                            // Only one valid integer in the range
                            if debug {
                                eprintln!("[MOD] Unique solution: {:?} = {}", term_id, first_valid);
                            }

                            // Check if there's a disequality that excludes this unique value
                            for &(diseq_lit, diseq_val) in &self.asserted {
                                if diseq_val {
                                    continue; // We want negated equalities (disequalities)
                                }
                                if debug {
                                    eprintln!(
                                        "[MOD] Checking asserted lit {:?} val={} for disequality",
                                        diseq_lit, diseq_val
                                    );
                                }
                                if let TermData::App(Symbol::Named(n), a) =
                                    self.terms.get(diseq_lit)
                                {
                                    if debug {
                                        eprintln!(
                                            "[MOD]   diseq structure: name={}, args={:?}",
                                            n, a
                                        );
                                    }
                                    if n == "=" && a.len() == 2 {
                                        // Check if this disequality involves term_id
                                        let (lhs, rhs) = (a[0], a[1]);
                                        if debug {
                                            eprintln!(
                                                "[MOD]   equality: lhs={:?}, rhs={:?}",
                                                lhs, rhs
                                            );
                                            eprintln!("[MOD]   looking for term_id={:?}", term_id);
                                        }
                                        let mut excluded_val = None;

                                        // Check if lhs is term_id and rhs is constant
                                        if lhs == *term_id {
                                            if let TermData::Const(Constant::Int(c)) =
                                                self.terms.get(rhs)
                                            {
                                                excluded_val = Some(c.clone());
                                            }
                                        }
                                        // Check if rhs is term_id and lhs is constant
                                        if rhs == *term_id {
                                            if let TermData::Const(Constant::Int(c)) =
                                                self.terms.get(lhs)
                                            {
                                                excluded_val = Some(c.clone());
                                            }
                                        }

                                        if let Some(excluded) = excluded_val {
                                            if excluded == first_valid {
                                                if debug {
                                                    eprintln!(
                                                        "[MOD] UNSAT: unique value {} excluded by disequality {:?}",
                                                        first_valid, diseq_lit
                                                    );
                                                }
                                                // Return conflict
                                                let mut conflict: Vec<TheoryLit> = self
                                                    .asserted
                                                    .iter()
                                                    .filter(|&&(_, v)| v) // Keep positive assertions
                                                    .map(|&(lit, val)| TheoryLit::new(lit, val))
                                                    .collect();
                                                conflict.push(TheoryLit::new(diseq_lit, false));
                                                return Some(conflict);
                                            }
                                        }
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

    /// Ceiling of a BigRational as BigInt
    fn ceil_bigint(val: &BigRational) -> BigInt {
        let (floor, ceil) = Self::floor_ceil_rational(val);
        if Self::is_integer(val) {
            floor
        } else {
            ceil
        }
    }

    /// Floor of a BigRational as BigInt
    fn floor_bigint(val: &BigRational) -> BigInt {
        let (floor, _) = Self::floor_ceil_rational(val);
        floor
    }

    /// Parse equality for Diophantine solver: lhs = rhs -> (coeffs, constant)
    fn parse_equality_for_dioph(
        &self,
        lhs: TermId,
        rhs: TermId,
        term_to_idx: &HashMap<TermId, usize>,
    ) -> (Vec<(usize, BigInt)>, BigInt) {
        let mut coeffs: HashMap<usize, BigInt> = HashMap::new();
        let mut constant = BigInt::zero();

        // Parse lhs with positive sign
        self.collect_dioph_terms(lhs, &BigInt::one(), &mut coeffs, &mut constant, term_to_idx);

        // Parse rhs with negative sign (move to LHS)
        self.collect_dioph_terms(
            rhs,
            &-BigInt::one(),
            &mut coeffs,
            &mut constant,
            term_to_idx,
        );

        // The equation is: Σ(coeff * var) - constant = 0
        // So: Σ(coeff * var) = constant
        constant = -constant;

        // Remove zero coefficients
        coeffs.retain(|_, v| !v.is_zero());

        let coeffs_vec: Vec<_> = coeffs.into_iter().collect();
        (coeffs_vec, constant)
    }

    /// Recursively collect linear terms for Diophantine solver
    fn collect_dioph_terms(
        &self,
        term: TermId,
        scale: &BigInt,
        coeffs: &mut HashMap<usize, BigInt>,
        constant: &mut BigInt,
        term_to_idx: &HashMap<TermId, usize>,
    ) {
        match self.terms.get(term) {
            TermData::Const(Constant::Int(n)) => {
                *constant += scale * n;
            }
            TermData::Const(Constant::Rational(r)) => {
                if r.0.denom().is_one() {
                    *constant += scale * r.0.numer();
                }
            }
            TermData::Var(_, _) => {
                if let Some(&idx) = term_to_idx.get(&term) {
                    *coeffs.entry(idx).or_insert_with(BigInt::zero) += scale;
                }
            }
            TermData::App(Symbol::Named(name), args) => {
                match name.as_str() {
                    "+" => {
                        for &arg in args {
                            self.collect_dioph_terms(arg, scale, coeffs, constant, term_to_idx);
                        }
                    }
                    "-" if args.len() == 1 => {
                        self.collect_dioph_terms(
                            args[0],
                            &-scale.clone(),
                            coeffs,
                            constant,
                            term_to_idx,
                        );
                    }
                    "-" if args.len() >= 2 => {
                        self.collect_dioph_terms(args[0], scale, coeffs, constant, term_to_idx);
                        for &arg in &args[1..] {
                            self.collect_dioph_terms(
                                arg,
                                &-scale.clone(),
                                coeffs,
                                constant,
                                term_to_idx,
                            );
                        }
                    }
                    "*" => {
                        let mut const_factor = BigInt::one();
                        let mut var_args = Vec::new();

                        for &arg in args {
                            if let Some(c) = self.extract_constant(arg) {
                                const_factor *= c;
                            } else {
                                var_args.push(arg);
                            }
                        }

                        let new_scale = scale * &const_factor;

                        if var_args.is_empty() {
                            *constant += &new_scale;
                        } else if var_args.len() == 1 {
                            self.collect_dioph_terms(
                                var_args[0],
                                &new_scale,
                                coeffs,
                                constant,
                                term_to_idx,
                            );
                        }
                        // Non-linear terms are ignored
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Add an integer bound as an equality constraint to LRA.
    /// This is used when Diophantine solving determines a variable's value.
    fn add_integer_bound(&mut self, term_id: TermId, value: &BigInt) {
        let debug = std::env::var("Z4_DEBUG_DIOPH").is_ok();

        // Ensure the variable is registered with LRA (may not be if it only
        // appeared in pure integer equalities handled by Diophantine solver)
        let lra_var = self.lra.ensure_var_registered(term_id);

        let rat_value = BigRational::from(value.clone());

        if debug {
            eprintln!(
                "[DIOPH] Adding bound: var {} (lra {}) = {}",
                term_id.0, lra_var, rat_value
            );
        }

        // Add as both lower and upper bound (equality)
        self.lra.add_direct_bound(lra_var, rat_value.clone(), true); // lower
        self.lra.add_direct_bound(lra_var, rat_value, false); // upper
    }

    /// Generate HNF (Hermite Normal Form) cuts from the current constraint set.
    ///
    /// HNF cuts work on the original constraint matrix, avoiding the slack variable
    /// issues that plague Gomory cuts. They're generated when:
    /// 1. The LRA relaxation is SAT with non-integer values
    /// 2. Gomory cuts failed (typically due to slack variables in the tableau)
    ///
    /// Returns true if cuts were generated and added.
    fn try_hnf_cuts(&mut self, fractional_var: TermId) -> bool {
        let debug = std::env::var("Z4_DEBUG_HNF").is_ok();

        if self.hnf_iterations >= self.max_hnf_iterations {
            if debug {
                eprintln!("[HNF] Max iterations reached");
            }
            return false;
        }

        // Build HNF cutter from asserted equality constraints
        let mut cutter = hnf::HnfCutter::new();

        // Map from term IDs to variable indices for the HNF matrix
        let mut term_to_idx: HashMap<TermId, usize> = HashMap::new();
        let mut idx_to_term: Vec<TermId> = Vec::new();

        // Deterministic ordering makes HNF behavior easier to debug.
        let mut int_vars: Vec<TermId> = self.integer_vars.iter().copied().collect();
        int_vars.sort_by_key(|t| t.0);

        for (idx, term) in int_vars.into_iter().enumerate() {
            term_to_idx.insert(term, idx);
            idx_to_term.push(term);
            cutter.register_var(idx);
        }

        // Collect constraints from asserted literals
        // Include:
        // 1. Explicit equalities (always tight)
        // 2. Tight inequalities (where current LP solution equals the bound)
        for &(literal, value) in &self.asserted {
            if !value {
                continue;
            }

            let TermData::App(Symbol::Named(name), args) = self.terms.get(literal) else {
                continue;
            };

            if args.len() != 2 {
                continue;
            }

            match name.as_str() {
                "=" => {
                    // Equality constraint - always tight
                    let (var_coeffs, constant) =
                        self.parse_equality_for_hnf(args[0], args[1], &term_to_idx);

                    if var_coeffs.is_empty() {
                        continue;
                    }

                    // Add as upper bound constraint (lhs = constant means lhs <= constant)
                    cutter.add_constraint(&var_coeffs, constant.clone(), true);
                    // And lower bound (lhs >= constant)
                    cutter.add_constraint(&var_coeffs, constant, false);
                }
                "<=" => {
                    // lhs <= rhs - check if tight at current solution
                    if self.is_constraint_tight(args[0], args[1]) {
                        let (var_coeffs, constant) =
                            self.parse_equality_for_hnf(args[0], args[1], &term_to_idx);

                        if !var_coeffs.is_empty() {
                            // Tight inequality: lhs = rhs at current solution
                            // Add as upper bound: Σ(coeff * var) <= constant
                            cutter.add_constraint(&var_coeffs, constant, true);
                            if debug {
                                eprintln!("[HNF] Added tight <= constraint");
                            }
                        }
                    }
                }
                ">=" => {
                    // lhs >= rhs - check if tight at current solution
                    if self.is_constraint_tight(args[0], args[1]) {
                        let (var_coeffs, constant) =
                            self.parse_equality_for_hnf(args[0], args[1], &term_to_idx);

                        if !var_coeffs.is_empty() {
                            // Tight inequality: lhs = rhs at current solution
                            // Add as lower bound: Σ(coeff * var) >= constant
                            cutter.add_constraint(&var_coeffs, constant, false);
                            if debug {
                                eprintln!("[HNF] Added tight >= constraint");
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if !cutter.has_constraints() {
            if debug {
                eprintln!("[HNF] No equality constraints to generate cuts from");
            }
            return false;
        }

        // Generate cuts
        let cuts = cutter.generate_cuts();

        if cuts.is_empty() {
            if debug {
                eprintln!("[HNF] No cuts generated");
            }
            return false;
        }

        if debug {
            eprintln!("[HNF] Generated {} cuts", cuts.len());
        }

        // Convert HNF cuts to LRA bounds
        let mut added_any = false;
        for cut in cuts {
            // Convert coefficients from term indices to TermIds first
            let mut term_coeffs_int: Vec<(TermId, BigInt)> = Vec::new();
            let mut lra_coeffs: Vec<(u32, BigRational)> = Vec::new();
            let mut term_coeffs: Vec<(TermId, BigRational)> = Vec::new();

            for (var_idx, coeff) in &cut.coeffs {
                let Some(&tid) = idx_to_term.get(*var_idx) else {
                    continue;
                };
                term_coeffs_int.push((tid, coeff.clone()));
                let rational_coeff = BigRational::from(coeff.clone());
                term_coeffs.push((tid, rational_coeff.clone()));
                if let Some(&lra_var) = self.lra.term_to_var().get(&tid) {
                    lra_coeffs.push((lra_var, rational_coeff));
                }
            }

            // Build key using TermIds (stable across theory instances)
            term_coeffs_int.sort_by_key(|(tid, _)| tid.0);
            let key = HnfCutKey {
                coeffs: term_coeffs_int,
                bound: cut.bound.clone(),
            };

            if self.seen_hnf_cuts.contains(&key) {
                continue;
            }

            if lra_coeffs.is_empty() {
                continue;
            }

            // Create and add the cut as a bound constraint
            let cut_bound = BigRational::from(cut.bound.clone());
            let gomory_cut = z4_lra::GomoryCut {
                coeffs: lra_coeffs,
                bound: cut_bound.clone(),
                is_lower: false, // HNF cuts are upper bounds (Σ coeff*x <= bound)
            };

            self.lra.add_gomory_cut(&gomory_cut, fractional_var);
            added_any = true;
            self.seen_hnf_cuts.insert(key);

            // Store the cut using TermIds for replay after LRA reset
            self.learned_cuts.push(StoredCut {
                coeffs: term_coeffs,
                bound: cut_bound,
                is_lower: false,
            });
        }

        if added_any {
            self.hnf_iterations += 1;
        }
        added_any
    }

    /// Parse an equality (lhs = rhs) into coefficient/constant form for HNF.
    /// Returns (coefficients, constant) where coefficients maps var_idx -> BigInt.
    fn parse_equality_for_hnf(
        &self,
        lhs: TermId,
        rhs: TermId,
        term_to_idx: &HashMap<TermId, usize>,
    ) -> (Vec<(usize, BigInt)>, BigInt) {
        let mut coeffs: HashMap<usize, BigInt> = HashMap::new();
        let mut constant = BigInt::zero();

        // Parse lhs with positive sign
        self.collect_hnf_terms(lhs, &BigInt::one(), &mut coeffs, &mut constant, term_to_idx);

        // Parse rhs with negative sign (move to LHS)
        self.collect_hnf_terms(
            rhs,
            &-BigInt::one(),
            &mut coeffs,
            &mut constant,
            term_to_idx,
        );

        // The equation is: Σ(coeff * var) - constant = 0
        // So: Σ(coeff * var) = constant
        constant = -constant;

        let coeffs_vec: Vec<_> = coeffs.into_iter().collect();
        (coeffs_vec, constant)
    }

    /// Recursively collect linear terms for HNF cut generation.
    fn collect_hnf_terms(
        &self,
        term: TermId,
        scale: &BigInt,
        coeffs: &mut HashMap<usize, BigInt>,
        constant: &mut BigInt,
        term_to_idx: &HashMap<TermId, usize>,
    ) {
        match self.terms.get(term) {
            TermData::Const(Constant::Int(n)) => {
                *constant += scale * n;
            }
            TermData::Const(Constant::Rational(r)) => {
                if r.0.denom().is_one() {
                    *constant += scale * r.0.numer();
                }
            }
            TermData::Var(_, _) => {
                if let Some(&idx) = term_to_idx.get(&term) {
                    *coeffs.entry(idx).or_insert_with(BigInt::zero) += scale;
                }
            }
            TermData::App(Symbol::Named(name), args) => {
                match name.as_str() {
                    "+" => {
                        for &arg in args {
                            self.collect_hnf_terms(arg, scale, coeffs, constant, term_to_idx);
                        }
                    }
                    "-" if args.len() == 1 => {
                        self.collect_hnf_terms(
                            args[0],
                            &-scale.clone(),
                            coeffs,
                            constant,
                            term_to_idx,
                        );
                    }
                    "-" if args.len() >= 2 => {
                        self.collect_hnf_terms(args[0], scale, coeffs, constant, term_to_idx);
                        for &arg in &args[1..] {
                            self.collect_hnf_terms(
                                arg,
                                &-scale.clone(),
                                coeffs,
                                constant,
                                term_to_idx,
                            );
                        }
                    }
                    "*" => {
                        let mut const_factor = BigInt::one();
                        let mut var_args = Vec::new();

                        for &arg in args {
                            if let Some(c) = self.extract_constant(arg) {
                                const_factor *= c;
                            } else {
                                var_args.push(arg);
                            }
                        }

                        let new_scale = scale * &const_factor;

                        if var_args.is_empty() {
                            *constant += &new_scale;
                        } else if var_args.len() == 1 {
                            self.collect_hnf_terms(
                                var_args[0],
                                &new_scale,
                                coeffs,
                                constant,
                                term_to_idx,
                            );
                        }
                        // Non-linear terms are ignored
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Check if an inequality constraint is tight at the current LP solution.
    ///
    /// An inequality `lhs <= rhs` (or `lhs >= rhs`) is tight when the current
    /// LP solution has `lhs = rhs`. This is useful for HNF cuts which need
    /// defining constraints of the current LP vertex.
    fn is_constraint_tight(&self, lhs: TermId, rhs: TermId) -> bool {
        // Evaluate both sides at current LP solution
        let lhs_val = self.evaluate_term_at_solution(lhs);
        let rhs_val = self.evaluate_term_at_solution(rhs);

        match (lhs_val, rhs_val) {
            (Some(l), Some(r)) => {
                // Check if they're equal (tight constraint)
                l == r
            }
            _ => false, // Can't evaluate - assume not tight
        }
    }

    /// Evaluate a term at the current LP solution.
    ///
    /// Returns the rational value of the term using current variable assignments
    /// from the LRA solver.
    fn evaluate_term_at_solution(&self, term: TermId) -> Option<BigRational> {
        match self.terms.get(term) {
            TermData::Const(Constant::Int(n)) => Some(BigRational::from(n.clone())),
            TermData::Const(Constant::Rational(r)) => Some(r.0.clone()),
            TermData::Var(_, _) => {
                // Look up current value from LRA solver
                self.lra.get_value(term)
            }
            TermData::App(Symbol::Named(name), args) => match name.as_str() {
                "+" => {
                    let mut sum = BigRational::zero();
                    for &arg in args {
                        sum += self.evaluate_term_at_solution(arg)?;
                    }
                    Some(sum)
                }
                "-" if args.len() == 1 => {
                    let val = self.evaluate_term_at_solution(args[0])?;
                    Some(-val)
                }
                "-" if args.len() >= 2 => {
                    let mut result = self.evaluate_term_at_solution(args[0])?;
                    for &arg in &args[1..] {
                        result -= self.evaluate_term_at_solution(arg)?;
                    }
                    Some(result)
                }
                "*" => {
                    let mut product = BigRational::one();
                    for &arg in args {
                        product *= self.evaluate_term_at_solution(arg)?;
                    }
                    Some(product)
                }
                _ => None,
            },
            _ => None,
        }
    }
}

impl TheorySolver for LiaSolver<'_> {
    fn assert_literal(&mut self, literal: TermId, value: bool) {
        // Unwrap NOT: NOT(inner)=true means inner=false
        let (term, val) = match self.terms.get(literal) {
            TermData::Not(inner) => (*inner, !value),
            _ => (literal, value),
        };

        // Collect integer variables from this literal
        self.collect_integer_vars(term);

        // Track assertion for conflict generation
        self.asserted.push((term, val));

        // Forward to LRA solver (which also handles NOT unwrapping)
        self.lra.assert_literal(literal, value);
    }

    fn check(&mut self) -> TheoryResult {
        let debug = std::env::var("Z4_DEBUG_LIA_CHECK").is_ok();

        // GCD test: quick check for integer infeasibility
        // For equations like 4x + 4y + 4z - 2w = 49, GCD(4,4,4,2)=2 doesn't divide 49
        if let Some(reasons) = self.gcd_test() {
            if debug {
                eprintln!("[LIA] GCD test detected UNSAT");
            }
            return TheoryResult::Unsat(reasons);
        }

        // Diophantine solver: for equality-dense problems, try variable elimination.
        // This can directly determine variable values or detect infeasibility,
        // avoiding the slow iterative HNF cuts approach.
        //
        // Only run when the equality atoms have changed. The equality_key comparison
        // handles caching across pop() operations - if branch-and-bound only changes
        // inequality bounds but not equalities, we skip redundant Diophantine work.
        let equality_key = self.equality_key();
        if debug {
            eprintln!(
                "[LIA] eq_key.len={} dioph_key.len={} eq={}",
                equality_key.len(),
                self.dioph_equality_key.len(),
                equality_key == self.dioph_equality_key
            );
        }
        let should_run_dioph = !equality_key.is_empty() && self.dioph_equality_key != equality_key;
        // Always update the key to avoid stale comparisons
        self.dioph_equality_key = equality_key;

        if should_run_dioph {
            // Try 2-variable equations with extended GCD first (handles linear_00 case)
            if let Some(reasons) = self.try_two_variable_solve() {
                if debug {
                    eprintln!("[LIA] 2-variable solver detected UNSAT");
                }
                return TheoryResult::Unsat(reasons);
            }

            // Try multi-equality systems
            if let Some(reasons) = self.try_diophantine_solve() {
                if debug {
                    eprintln!("[LIA] Diophantine solver detected UNSAT");
                }
                return TheoryResult::Unsat(reasons);
            }
        }

        // Reset iteration counters at the start of each check.
        // These limit how much work we do inside a single theory check call.
        self.gomory_iterations = 0;
        self.hnf_iterations = 0;

        loop {
            // First check the LRA relaxation
            let lra_result = self.lra.check();

            if debug {
                eprintln!(
                    "[LIA] LRA check result: {:?}, gomory_iter={}, hnf_iter={}",
                    lra_result, self.gomory_iterations, self.hnf_iterations
                );
            }

            match lra_result {
                TheoryResult::Unsat(reasons) => {
                    // If LRA is UNSAT, so is LIA
                    return TheoryResult::Unsat(reasons);
                }
                TheoryResult::Unknown => {
                    if debug {
                        eprintln!("[LIA] LRA returned Unknown, propagating");
                    }
                    return TheoryResult::Unknown;
                }
                TheoryResult::NeedSplit(split) => {
                    // Forward split requests from LRA (shouldn't happen, but handle it)
                    return TheoryResult::NeedSplit(split);
                }
                TheoryResult::NeedDisequlitySplit(split) => {
                    // Before forwarding a disequality split, check if modular constraints
                    // make the excluded value the only valid integer.
                    // If so, the disequality makes the formula UNSAT.
                    if let Some(reasons) = self.check_disequality_vs_modular(&split) {
                        if debug {
                            eprintln!("[LIA] Disequality conflicts with modular constraint");
                        }
                        return TheoryResult::Unsat(reasons);
                    }
                    // Forward disequality split requests from LRA
                    return TheoryResult::NeedDisequlitySplit(split);
                }
                TheoryResult::NeedExpressionSplit(split) => {
                    // Forward expression split requests from LRA (multi-variable disequalities)
                    return TheoryResult::NeedExpressionSplit(split);
                }
                TheoryResult::Sat => {
                    if let Some(reasons) = self.check_integer_bounds_conflict() {
                        return TheoryResult::Unsat(reasons);
                    }

                    // Check modular constraints from equalities against bounds.
                    // This detects cases like r = 2*x - 2*y with 0 <= r < 2,
                    // where r must be a multiple of 2 but the bounds only allow 0 or 1.
                    if let Some(reasons) = self.check_single_equality_modular_constraints() {
                        if debug {
                            eprintln!("[LIA] Modular constraint detected UNSAT");
                        }
                        return TheoryResult::Unsat(reasons);
                    }

                    // Also check modular constraints from Diophantine substitutions
                    if let Some(reasons) = self.check_modular_constraint_conflict() {
                        if debug {
                            eprintln!("[LIA] Dioph modular constraint detected UNSAT");
                        }
                        return TheoryResult::Unsat(reasons);
                    }

                    // Try direct lattice enumeration for equality-dense systems.
                    // This uses rational Gaussian elimination + enumeration to directly
                    // find integer solutions, avoiding expensive branch-and-bound.
                    // Must be called after LRA SAT so that bounds are available.
                    if self.gomory_iterations == 0 && self.hnf_iterations == 0 {
                        // Only try on first iteration to avoid redundant work
                        if let Some(reasons) = self.try_direct_enumeration() {
                            if debug {
                                eprintln!("[LIA] Direct enumeration detected UNSAT");
                            }
                            return TheoryResult::Unsat(reasons);
                        }
                    }

                    // LRA is SAT - check integer constraints
                    if let Some((var, value)) = self.check_integer_constraints() {
                        // Found a variable with non-integer value.
                        // Try techniques in order: Patching, Gomory cuts, HNF cuts, Branch

                        // 0. Try patching (Z3's technique to avoid branching entirely)
                        // This adjusts non-basic integer variables to make basic vars integral
                        if self.try_patching() {
                            if debug {
                                eprintln!("[LIA] Patching succeeded, re-checking");
                            }
                            continue;
                        }

                        // 1. Try Gomory cuts (fast, but limited by slack variables)
                        if self.gomory_iterations < self.max_gomory_iterations {
                            let cuts = self.lra.generate_gomory_cuts(&self.integer_vars);

                            if debug {
                                eprintln!(
                                    "[LIA] Generated {} Gomory cuts (iter {})",
                                    cuts.len(),
                                    self.gomory_iterations
                                );
                            }

                            if !cuts.is_empty() {
                                for cut in &cuts {
                                    self.lra.add_gomory_cut(cut, var);
                                }
                                self.gomory_iterations += 1;
                                continue;
                            }
                        }

                        // 2. Try HNF cuts (works on original constraints, avoids slack var issues)
                        // HNF cuts use tight inequalities (active at current LP solution) along
                        // with explicit equalities.
                        //
                        // For equality-dense problems (many equalities relative to variables),
                        // the solution space is highly constrained. In such cases, we allow
                        // more HNF iterations since the cuts can be very effective.
                        //
                        // With k equalities in n variables, the solution space has (n-k) free
                        // dimensions (assuming the equalities are independent). We consider
                        // a problem "equality-dense" when the free dimensions are at most 1,
                        // i.e., when k >= n-1.
                        let num_equalities = self.count_equalities();
                        let num_vars = self.integer_vars.len();
                        let is_equality_dense = num_vars > 0 && num_equalities + 1 >= num_vars;

                        // Allow more iterations for equality-dense problems
                        // With k equalities in n variables, we may need O(k) iterations
                        // to tighten all dimensions of the lattice.
                        let max_hnf_per_check = if is_equality_dense {
                            20 // More iterations when equalities nearly determine the solution
                        } else {
                            2 // Standard: try twice (second iteration might help after cuts change LP)
                        };

                        // Track whether we've made any HNF progress in this check() call
                        let pre_hnf_iter = self.hnf_iterations;

                        while self.hnf_iterations < max_hnf_per_check {
                            if debug {
                                eprintln!(
                                    "[LIA] Trying HNF cuts (iter {}/{}, {} equalities, {} vars, dense={})",
                                    self.hnf_iterations, max_hnf_per_check,
                                    num_equalities, num_vars, is_equality_dense
                                );
                            }
                            if self.try_hnf_cuts(var) {
                                if debug {
                                    eprintln!(
                                        "[LIA] HNF cuts generated, continuing inner HNF loop"
                                    );
                                }
                                // Check if we've hit iteration limit inside the while loop
                                continue;
                            }
                            // No new cuts generated - break out of HNF loop
                            break;
                        }

                        // If we generated any HNF cuts, re-check the LRA solution
                        if self.hnf_iterations > pre_hnf_iter {
                            if debug {
                                eprintln!(
                                    "[LIA] Generated {} HNF cuts total, re-checking LRA",
                                    self.hnf_iterations - pre_hnf_iter
                                );
                            }
                            continue;
                        }

                        // 3. Fall back to branch-and-bound
                        if debug {
                            eprintln!(
                                "[LIA] Falling back to branch-and-bound (gomory={}, hnf={})",
                                self.gomory_iterations, self.hnf_iterations
                            );
                        }
                        let split = self.create_split_request(var, value);
                        return TheoryResult::NeedSplit(split);
                    } else {
                        // All integer constraints satisfied
                        return TheoryResult::Sat;
                    }
                }
            }
        }
    }

    fn propagate(&mut self) -> Vec<TheoryPropagation> {
        // Forward propagations from LRA
        self.lra.propagate()
    }

    fn push(&mut self) {
        self.scopes.push(self.asserted.len());
        self.lra.push();
    }

    fn pop(&mut self) {
        if let Some(mark) = self.scopes.pop() {
            self.asserted.truncate(mark);
            self.lra.pop();
            // Note: We preserve dioph_equality_key, dioph_safe_dependent_vars, and
            // dioph_cached_substitutions across pop(). The equality structure doesn't
            // change during backtracking (only inequality bounds change), so this
            // information remains valid. The equality_key comparison in check() will
            // naturally detect if the equality structure changes in a different context.
        }
    }

    fn reset(&mut self) {
        self.lra.reset();
        self.integer_vars.clear();
        self.asserted.clear();
        self.scopes.clear();
        self.gomory_iterations = 0;
        self.hnf_iterations = 0;
        self.seen_hnf_cuts.clear();
        self.learned_cuts.clear();
        self.dioph_equality_key.clear();
        self.dioph_safe_dependent_vars.clear();
    }

    fn soft_reset(&mut self) {
        // Use clear_assertions which preserves learned HNF cuts
        self.clear_assertions();
    }
}

// Additional methods for incremental solving (not part of TheorySolver trait)
impl LiaSolver<'_> {
    /// Clear assertions but preserve learned cuts.
    ///
    /// Use this between SAT model iterations in DPLL(T) to retain HNF cuts
    /// that are globally valid (derived from the original constraint matrix).
    pub fn clear_assertions(&mut self) {
        self.lra.reset();
        self.integer_vars.clear();
        self.asserted.clear();
        self.scopes.clear();
        self.gomory_iterations = 0;
        self.hnf_iterations = 0;
        // IMPORTANT: Preserve dioph_equality_key, dioph_safe_dependent_vars, and
        // dioph_cached_substitutions. These are derived from the equality structure
        // which typically doesn't change across soft resets (only inequality bounds
        // change during branch-and-bound). If the equality structure changes, the
        // equality_key comparison in check() will naturally invalidate the cache.
        //
        // Preserving these allows:
        // 1. Skipping redundant Diophantine solving
        // 2. Using cached branching hints (safe_dependent_vars)
        // 3. Propagating bounds through cached substitutions
        //
        // Also preserve seen_hnf_cuts and learned_cuts - they're globally valid
    }

    /// Get the learned cuts and seen cut keys for external storage.
    ///
    /// Use this to preserve cuts across theory instances when the theory
    /// must be dropped temporarily (e.g., to allow mutable term store access).
    /// Returns (learned_cuts, seen_hnf_cut_keys).
    pub fn take_learned_state(&mut self) -> (Vec<StoredCut>, HashSet<HnfCutKey>) {
        let debug = std::env::var("Z4_DEBUG_HNF").is_ok();
        if debug {
            eprintln!(
                "[HNF] Taking {} learned cuts, {} seen keys",
                self.learned_cuts.len(),
                self.seen_hnf_cuts.len()
            );
        }
        (
            std::mem::take(&mut self.learned_cuts),
            std::mem::take(&mut self.seen_hnf_cuts),
        )
    }

    /// Import previously learned cuts and seen cut keys.
    ///
    /// Use this to restore state from a previous theory instance.
    pub fn import_learned_state(&mut self, cuts: Vec<StoredCut>, seen: HashSet<HnfCutKey>) {
        let debug = std::env::var("Z4_DEBUG_HNF").is_ok();
        if debug {
            eprintln!(
                "[HNF] Importing {} learned cuts, {} seen keys",
                cuts.len(),
                seen.len()
            );
        }
        self.learned_cuts = cuts;
        self.seen_hnf_cuts = seen;
    }

    /// Replay learned cuts into the LRA solver.
    ///
    /// Call this after asserting new literals to restore previously learned cuts.
    /// Cuts that reference unknown variables are skipped.
    pub fn replay_learned_cuts(&mut self) {
        let debug = std::env::var("Z4_DEBUG_HNF").is_ok();
        if debug && !self.learned_cuts.is_empty() {
            eprintln!(
                "[HNF] Attempting to replay {} learned cuts, integer_vars={}, lra_vars={}",
                self.learned_cuts.len(),
                self.integer_vars.len(),
                self.lra.term_to_var().len()
            );
        }
        let mut replayed = 0;
        let mut skipped_no_var = 0;
        let mut skipped_no_int = 0;

        for cut in &self.learned_cuts {
            // Convert TermIds to LRA variable IDs
            let mut lra_coeffs: Vec<(u32, BigRational)> = Vec::new();
            let mut all_valid = true;

            for (term_id, coeff) in &cut.coeffs {
                if let Some(&lra_var) = self.lra.term_to_var().get(term_id) {
                    lra_coeffs.push((lra_var, coeff.clone()));
                } else {
                    // Variable not in current LRA state - skip this cut
                    all_valid = false;
                    skipped_no_var += 1;
                    break;
                }
            }

            if all_valid && !lra_coeffs.is_empty() {
                // Re-add the cut as a bound
                // Use a dummy fractional_var since we're just adding a global bound
                let gomory_cut = z4_lra::GomoryCut {
                    coeffs: lra_coeffs,
                    bound: cut.bound.clone(),
                    is_lower: cut.is_lower,
                };
                // Find any integer variable to use as reference (pick min for determinism)
                if let Some(&ref_var) = self.integer_vars.iter().min() {
                    self.lra.add_gomory_cut(&gomory_cut, ref_var);
                    replayed += 1;
                } else {
                    skipped_no_int += 1;
                }
            }
        }

        if debug {
            if replayed > 0 {
                eprintln!("[HNF] Replayed {} learned cuts", replayed);
            }
            if skipped_no_var > 0 || skipped_no_int > 0 {
                eprintln!(
                    "[HNF] Skipped cuts: {} no LRA var, {} no int var",
                    skipped_no_var, skipped_no_int
                );
            }
        }
    }

    /// Get the Diophantine solver state for external storage.
    ///
    /// Use this to preserve Diophantine analysis across theory instances when the
    /// theory must be dropped temporarily (e.g., for mutable term store access).
    ///
    /// The Diophantine solver analyzes equality structure to:
    /// 1. Eliminate variables via substitution
    /// 2. Identify safe dependent variables (poor branching candidates)
    /// 3. Propagate bounds through substitutions
    ///
    /// Since the equality structure typically doesn't change during lazy DPLL(T)
    /// (only inequality bounds change), preserving this state avoids redundant
    /// Diophantine analysis.
    pub fn take_dioph_state(&mut self) -> DiophState {
        let debug = std::env::var("Z4_DEBUG_DIOPH").is_ok();
        if debug {
            eprintln!(
                "[DIOPH] Taking state: {} equalities, {} safe vars, {} substitutions",
                self.dioph_equality_key.len(),
                self.dioph_safe_dependent_vars.len(),
                self.dioph_cached_substitutions.len()
            );
        }
        (
            std::mem::take(&mut self.dioph_equality_key),
            std::mem::take(&mut self.dioph_safe_dependent_vars),
            std::mem::take(&mut self.dioph_cached_substitutions),
        )
    }

    /// Import previously computed Diophantine solver state.
    ///
    /// Use this to restore state from a previous theory instance.
    pub fn import_dioph_state(&mut self, state: DiophState) {
        let (equality_key, safe_dependent_vars, cached_substitutions) = state;
        let debug = std::env::var("Z4_DEBUG_DIOPH").is_ok();
        if debug {
            eprintln!(
                "[DIOPH] Importing state: {} equalities, {} safe vars, {} substitutions",
                equality_key.len(),
                safe_dependent_vars.len(),
                cached_substitutions.len()
            );
        }
        self.dioph_equality_key = equality_key;
        self.dioph_safe_dependent_vars = safe_dependent_vars;
        self.dioph_cached_substitutions = cached_substitutions;
    }
}

// ============================================================================
// Kani Verification Harnesses
// ============================================================================
//
// These proofs verify the core invariants of the LIA (Linear Integer Arithmetic) solver:
// 1. is_integer correctness: returns true iff denominator is 1
// 2. floor_ceil_rational correctness: floor <= value <= ceil
// 3. Split request validity: floor < value < ceil for non-integers
// 4. Push/pop state consistency

#[cfg(kani)]
mod verification {
    use super::*;

    // ========================================================================
    // Integer Detection Invariants
    // ========================================================================

    /// is_integer returns true for whole numbers
    #[kani::proof]
    fn proof_is_integer_for_whole_numbers() {
        let n: i32 = kani::any();
        kani::assume(n > -1000 && n < 1000);

        let rat = BigRational::from(BigInt::from(n));
        assert!(LiaSolver::is_integer(&rat), "Whole numbers are integers");
    }

    /// is_integer returns false for proper fractions
    #[kani::proof]
    fn proof_is_integer_for_fractions() {
        let numer: i32 = kani::any();
        let denom: i32 = kani::any();
        kani::assume(numer > -100 && numer < 100);
        kani::assume(denom > 1 && denom < 10);
        kani::assume(numer % denom != 0); // Not divisible

        let rat = BigRational::new(BigInt::from(numer), BigInt::from(denom));
        assert!(
            !LiaSolver::is_integer(&rat),
            "Proper fractions are not integers"
        );
    }

    /// is_integer returns true when numerator is divisible by denominator
    #[kani::proof]
    fn proof_is_integer_when_divisible() {
        let k: i32 = kani::any();
        let d: i32 = kani::any();
        kani::assume(k > -50 && k < 50);
        kani::assume(d > 0 && d < 10);

        // k*d / d = k, which is an integer
        let rat = BigRational::new(BigInt::from(k * d), BigInt::from(d));
        assert!(LiaSolver::is_integer(&rat), "k*d/d should be integer k");
    }

    // ========================================================================
    // Floor/Ceil Invariants
    // ========================================================================

    /// floor_ceil_rational: floor <= value <= ceil
    #[kani::proof]
    fn proof_floor_ceil_bounds() {
        let numer: i32 = kani::any();
        let denom: i32 = kani::any();
        kani::assume(numer > -100 && numer < 100);
        kani::assume(denom > 0 && denom < 10);

        let rat = BigRational::new(BigInt::from(numer), BigInt::from(denom));
        let (floor, ceil) = LiaSolver::floor_ceil_rational(&rat);

        // Convert floor/ceil to rationals for comparison
        let floor_rat = BigRational::from(floor.clone());
        let ceil_rat = BigRational::from(ceil.clone());

        assert!(floor_rat <= rat, "floor <= value");
        assert!(rat <= ceil_rat, "value <= ceil");
    }

    /// floor_ceil_rational: floor + 1 >= ceil (they're adjacent or equal)
    #[kani::proof]
    fn proof_floor_ceil_adjacent() {
        let numer: i32 = kani::any();
        let denom: i32 = kani::any();
        kani::assume(numer > -100 && numer < 100);
        kani::assume(denom > 0 && denom < 10);

        let rat = BigRational::new(BigInt::from(numer), BigInt::from(denom));
        let (floor, ceil) = LiaSolver::floor_ceil_rational(&rat);

        let diff = &ceil - &floor;
        assert!(diff <= BigInt::one(), "ceil - floor <= 1");
        assert!(diff >= BigInt::zero(), "ceil >= floor");
    }

    /// floor_ceil_rational: for integers, floor == ceil == value
    #[kani::proof]
    fn proof_floor_ceil_for_integers() {
        let n: i32 = kani::any();
        kani::assume(n > -100 && n < 100);

        let rat = BigRational::from(BigInt::from(n));
        let (floor, ceil) = LiaSolver::floor_ceil_rational(&rat);

        let expected = BigInt::from(n);
        assert!(floor == expected, "floor of integer is itself");
        assert!(ceil == expected, "ceil of integer is itself");
    }

    /// floor_ceil_rational: for non-integers, floor < value < ceil
    #[kani::proof]
    fn proof_floor_ceil_for_non_integers() {
        let numer: i32 = kani::any();
        let denom: i32 = kani::any();
        kani::assume(numer > -50 && numer < 50);
        kani::assume(denom > 1 && denom < 5);
        kani::assume(numer % denom != 0); // Not an integer

        let rat = BigRational::new(BigInt::from(numer), BigInt::from(denom));
        let (floor, ceil) = LiaSolver::floor_ceil_rational(&rat);

        let floor_rat = BigRational::from(floor.clone());
        let ceil_rat = BigRational::from(ceil.clone());

        assert!(floor_rat < rat, "floor < value for non-integers");
        assert!(rat < ceil_rat, "value < ceil for non-integers");
        assert!(
            ceil == floor + BigInt::one(),
            "ceil = floor + 1 for non-integers"
        );
    }

    /// floor_ceil_rational: negative values handled correctly
    #[kani::proof]
    fn proof_floor_ceil_negative() {
        // Test -1/2 = -0.5: floor should be -1, ceil should be 0
        let rat = BigRational::new(BigInt::from(-1), BigInt::from(2));
        let (floor, ceil) = LiaSolver::floor_ceil_rational(&rat);

        assert!(floor == BigInt::from(-1), "floor(-0.5) = -1");
        assert!(ceil == BigInt::from(0), "ceil(-0.5) = 0");

        // Test -3/2 = -1.5: floor should be -2, ceil should be -1
        let rat2 = BigRational::new(BigInt::from(-3), BigInt::from(2));
        let (floor2, ceil2) = LiaSolver::floor_ceil_rational(&rat2);

        assert!(floor2 == BigInt::from(-2), "floor(-1.5) = -2");
        assert!(ceil2 == BigInt::from(-1), "ceil(-1.5) = -1");
    }

    // ========================================================================
    // Split Request Invariants
    // ========================================================================

    /// Split request creates valid floor/ceil
    #[kani::proof]
    fn proof_split_request_validity() {
        let terms = z4_core::term::TermStore::new();
        let solver = LiaSolver::new(&terms);

        let numer: i32 = kani::any();
        let denom: i32 = kani::any();
        kani::assume(numer > -50 && numer < 50);
        kani::assume(denom > 1 && denom < 5);
        kani::assume(numer % denom != 0); // Non-integer

        let value = BigRational::new(BigInt::from(numer), BigInt::from(denom));
        let dummy_term_id = z4_core::term::TermId(0);

        let split = solver.create_split_request(dummy_term_id, value.clone());

        // Verify floor < value < ceil
        let floor_rat = BigRational::from(split.floor.clone());
        let ceil_rat = BigRational::from(split.ceil.clone());

        assert!(floor_rat < value, "Split floor < value");
        assert!(value < ceil_rat, "value < Split ceil");
        assert!(
            split.ceil == split.floor + BigInt::one(),
            "ceil = floor + 1"
        );
    }

    // ========================================================================
    // Solver State Invariants
    // ========================================================================

    /// Push/pop maintains scope stack correctly
    #[kani::proof]
    fn proof_push_pop_scope_depth() {
        let terms = z4_core::term::TermStore::new();
        let mut solver = LiaSolver::new(&terms);

        assert!(solver.scopes.is_empty(), "Initially no scopes");

        solver.push();
        assert!(solver.scopes.len() == 1, "Push adds scope");

        solver.push();
        assert!(solver.scopes.len() == 2, "Second push adds scope");

        solver.pop();
        assert!(solver.scopes.len() == 1, "Pop removes scope");

        solver.pop();
        assert!(solver.scopes.is_empty(), "Final pop returns to empty");
    }

    /// Pop on empty scopes is safe (no-op)
    #[kani::proof]
    fn proof_pop_empty_is_safe() {
        let terms = z4_core::term::TermStore::new();
        let mut solver = LiaSolver::new(&terms);

        solver.pop();
        assert!(solver.scopes.is_empty(), "Pop on empty is no-op");
    }

    /// Reset clears all state
    #[kani::proof]
    fn proof_reset_clears_state() {
        let terms = z4_core::term::TermStore::new();
        let mut solver = LiaSolver::new(&terms);

        // Add some state
        solver.push();
        solver.integer_vars.insert(z4_core::term::TermId(42));
        solver.asserted.push((z4_core::term::TermId(1), true));

        solver.reset();

        assert!(solver.integer_vars.is_empty(), "Reset clears integer_vars");
        assert!(solver.asserted.is_empty(), "Reset clears asserted");
        assert!(solver.scopes.is_empty(), "Reset clears scopes");
    }

    /// Register integer var adds to set
    #[kani::proof]
    fn proof_register_integer_var() {
        let terms = z4_core::term::TermStore::new();
        let mut solver = LiaSolver::new(&terms);

        let term = z4_core::term::TermId(5);
        assert!(
            !solver.integer_vars.contains(&term),
            "Not initially registered"
        );

        solver.register_integer_var(term);
        assert!(solver.integer_vars.contains(&term), "Term is registered");

        // Registering twice is idempotent
        solver.register_integer_var(term);
        assert!(
            solver.integer_vars.len() == 1,
            "Duplicate registration is idempotent"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use z4_core::term::TermStore;

    #[test]
    fn test_lia_solver_trivial_sat() {
        let terms = TermStore::new();
        let mut solver = LiaSolver::new(&terms);

        // Empty problem is SAT
        assert!(matches!(solver.check(), TheoryResult::Sat));
    }

    #[test]
    fn test_lia_solver_integer_bound() {
        let mut terms = TermStore::new();

        // x <= 5 where x is an integer
        let x = terms.mk_var("x", Sort::Int);
        let five = terms.mk_int(BigInt::from(5));
        let atom = terms.mk_le(x, five);

        let mut solver = LiaSolver::new(&terms);
        solver.assert_literal(atom, true);

        // Should be SAT (x can be any integer <= 5)
        assert!(matches!(solver.check(), TheoryResult::Sat));
    }

    #[test]
    fn test_lia_solver_conflicting_bounds() {
        let mut terms = TermStore::new();

        // x >= 10 and x <= 5 should be UNSAT
        let x = terms.mk_var("x", Sort::Int);
        let five = terms.mk_int(BigInt::from(5));
        let ten = terms.mk_int(BigInt::from(10));

        let le_atom = terms.mk_le(x, five); // x <= 5
        let ge_atom = terms.mk_ge(x, ten); // x >= 10

        let mut solver = LiaSolver::new(&terms);
        solver.assert_literal(le_atom, true);
        solver.assert_literal(ge_atom, true);

        let result = solver.check();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_lia_solver_integer_infeasible() {
        let mut terms = TermStore::new();

        // x > 5 and x < 6 where x is integer - UNSAT (no integer between 5 and 6)
        let x = terms.mk_var("x", Sort::Int);
        let five = terms.mk_int(BigInt::from(5));
        let six = terms.mk_int(BigInt::from(6));

        let gt_atom = terms.mk_gt(x, five); // x > 5
        let lt_atom = terms.mk_lt(x, six); // x < 6

        let mut solver = LiaSolver::new(&terms);
        solver.assert_literal(gt_atom, true);
        solver.assert_literal(lt_atom, true);

        // LRA would say SAT with x = 5.5, but LIA should reject
        let result = solver.check();
        assert!(matches!(result, TheoryResult::Unsat(_)), "{result:?}");
    }

    #[test]
    fn test_lia_solver_integer_feasible() {
        let mut terms = TermStore::new();

        // x >= 5 and x <= 6 where x is integer - SAT (x can be 5 or 6)
        let x = terms.mk_var("x", Sort::Int);
        let five = terms.mk_int(BigInt::from(5));
        let six = terms.mk_int(BigInt::from(6));

        let ge_atom = terms.mk_ge(x, five); // x >= 5
        let le_atom = terms.mk_le(x, six); // x <= 6

        let mut solver = LiaSolver::new(&terms);
        solver.assert_literal(ge_atom, true);
        solver.assert_literal(le_atom, true);

        let result = solver.check();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_lia_solver_linear_constraint() {
        let mut terms = TermStore::new();

        // 2*x >= 5 where x is integer
        // LRA: x >= 2.5, so x = 2.5 is valid
        // LIA: x must be integer, so x >= 3
        let x = terms.mk_var("x", Sort::Int);
        let two = terms.mk_int(BigInt::from(2));
        let five = terms.mk_int(BigInt::from(5));

        let scaled = terms.mk_mul(vec![two, x]);
        let ge_atom = terms.mk_ge(scaled, five); // 2*x >= 5

        let mut solver = LiaSolver::new(&terms);
        solver.assert_literal(ge_atom, true);

        // LRA gives x = 2.5 which is non-integer, so LIA returns NeedSplit
        // to request a splitting lemma (x <= 2) OR (x >= 3)
        // This is the correct branch-and-bound behavior - the DPLL layer
        // should handle the split and eventually find x = 3.
        let result = solver.check();
        assert!(matches!(
            result,
            TheoryResult::Sat | TheoryResult::NeedSplit(_)
        ));
    }

    #[test]
    fn test_parse_linear_expr_with_negated_constant_mul() {
        let mut terms = TermStore::new();

        let x = terms.mk_var("x", Sort::Int);
        let zero = terms.mk_int(BigInt::from(0));
        let one = terms.mk_int(BigInt::from(1));
        let neg_one = terms.mk_neg(one); // Represents `(- 1)`
        let neg_x = terms.mk_mul(vec![neg_one, x]); // Represents `(* (- 1) x)`

        let solver = LiaSolver::new(&terms);
        let (coeffs, constant) = solver.parse_linear_expr_with_vars(neg_x, zero);

        assert_eq!(constant, BigInt::from(0));
        assert_eq!(coeffs.len(), 1);
        assert_eq!(coeffs.get(&x), Some(&BigInt::from(-1)));
    }

    #[test]
    fn test_lia_solver_push_pop() {
        let mut terms = TermStore::new();

        let x = terms.mk_var("x", Sort::Int);
        let three = terms.mk_int(BigInt::from(3));
        let five = terms.mk_int(BigInt::from(5));

        let ge_atom = terms.mk_ge(x, three); // x >= 3
        let le_atom = terms.mk_le(x, five); // x <= 5

        let mut solver = LiaSolver::new(&terms);
        solver.assert_literal(ge_atom, true);

        assert!(matches!(solver.check(), TheoryResult::Sat));

        // Push and add constraint
        solver.push();
        solver.assert_literal(le_atom, true);

        assert!(matches!(solver.check(), TheoryResult::Sat));

        // Pop should restore previous state
        solver.pop();
        // After pop, only ge_atom is asserted
        solver.reset();
        solver.assert_literal(ge_atom, true);
        assert!(matches!(solver.check(), TheoryResult::Sat));
    }

    #[test]
    fn test_is_integer() {
        assert!(LiaSolver::is_integer(&BigRational::from(BigInt::from(5))));
        assert!(LiaSolver::is_integer(&BigRational::from(BigInt::from(0))));
        assert!(LiaSolver::is_integer(&BigRational::from(BigInt::from(-10))));

        // 5/2 = 2.5 is not an integer
        let half = BigRational::new(BigInt::from(5), BigInt::from(2));
        assert!(!LiaSolver::is_integer(&half));

        // 6/3 = 2 is an integer (should simplify)
        let two = BigRational::new(BigInt::from(6), BigInt::from(3));
        assert!(LiaSolver::is_integer(&two));
    }

    #[test]
    fn test_split_request_floor_ceil_negative() {
        let mut terms = TermStore::new();
        let x = terms.mk_var("x", Sort::Int);
        let solver = LiaSolver::new(&terms);

        let s = solver.create_split_request(x, BigRational::new(BigInt::from(-1), BigInt::from(2)));
        assert_eq!(s.floor, BigInt::from(-1));
        assert_eq!(s.ceil, BigInt::from(0));

        let s = solver.create_split_request(x, BigRational::new(BigInt::from(-3), BigInt::from(2)));
        assert_eq!(s.floor, BigInt::from(-2));
        assert_eq!(s.ceil, BigInt::from(-1));

        let s = solver.create_split_request(x, BigRational::new(BigInt::from(7), BigInt::from(2)));
        assert_eq!(s.floor, BigInt::from(3));
        assert_eq!(s.ceil, BigInt::from(4));
    }

    #[test]
    fn test_lia_model_extraction() {
        let mut terms = TermStore::new();

        // x = 5 where x is integer
        let x = terms.mk_var("x", Sort::Int);
        let five = terms.mk_int(BigInt::from(5));
        let eq_atom = terms.mk_eq(x, five);

        let mut solver = LiaSolver::new(&terms);
        solver.assert_literal(eq_atom, true);

        let result = solver.check();
        assert!(matches!(result, TheoryResult::Sat));

        // Extract and verify model
        if let Some(model) = solver.extract_model() {
            if let Some(val) = model.values.get(&x) {
                assert_eq!(*val, BigInt::from(5));
            }
        }
    }

    #[test]
    fn test_lia_solver_two_var_equality_not_immediately_unsat() {
        let mut terms = TermStore::new();

        // 4*x + 3*y = 70, x >= 0, y >= 0, x <= 41 is satisfiable over integers.
        let x = terms.mk_var("x", Sort::Int);
        let y = terms.mk_var("y", Sort::Int);
        let four = terms.mk_int(BigInt::from(4));
        let three = terms.mk_int(BigInt::from(3));
        let seventy = terms.mk_int(BigInt::from(70));
        let zero = terms.mk_int(BigInt::from(0));
        let forty_one = terms.mk_int(BigInt::from(41));

        let four_x = terms.mk_mul(vec![four, x]);
        let three_y = terms.mk_mul(vec![three, y]);
        let lhs = terms.mk_add(vec![four_x, three_y]);
        let eq = terms.mk_eq(lhs, seventy);
        let x_ge = terms.mk_ge(x, zero);
        let y_ge = terms.mk_ge(y, zero);
        let x_le = terms.mk_le(x, forty_one);

        let mut solver = LiaSolver::new(&terms);
        solver.assert_literal(eq, true);
        solver.assert_literal(x_ge, true);
        solver.assert_literal(y_ge, true);
        solver.assert_literal(x_le, true);

        let result = solver.check();
        assert!(
            matches!(result, TheoryResult::Sat | TheoryResult::NeedSplit(_)),
            "{result:?}"
        );
    }

    #[test]
    fn test_gcd_test_unsat() {
        let mut terms = TermStore::new();

        // 4*x + 4*y + 4*z - 2*w = 49
        // GCD(4, 4, 4, 2) = 2, but 2 does not divide 49, so UNSAT
        let x = terms.mk_var("x", Sort::Int);
        let y = terms.mk_var("y", Sort::Int);
        let z = terms.mk_var("z", Sort::Int);
        let w = terms.mk_var("w", Sort::Int);

        let four = terms.mk_int(BigInt::from(4));
        let minus_two = terms.mk_int(BigInt::from(-2));
        let forty_nine = terms.mk_int(BigInt::from(49));

        let four_x = terms.mk_mul(vec![four, x]);
        let four_y = terms.mk_mul(vec![four, y]);
        let four_z = terms.mk_mul(vec![four, z]);
        let minus_two_w = terms.mk_mul(vec![minus_two, w]);

        let lhs = terms.mk_add(vec![four_x, four_y, four_z, minus_two_w]);
        let eq = terms.mk_eq(lhs, forty_nine);

        let mut solver = LiaSolver::new(&terms);
        solver.assert_literal(eq, true);

        let result = solver.check();
        assert!(
            matches!(result, TheoryResult::Unsat(_)),
            "GCD test should detect UNSAT: {result:?}"
        );
    }

    #[test]
    fn test_gcd_test_sat() {
        let mut terms = TermStore::new();

        // 2*x + 4*y = 10
        // GCD(2, 4) = 2, and 2 divides 10, so may be SAT
        let x = terms.mk_var("x", Sort::Int);
        let y = terms.mk_var("y", Sort::Int);

        let two = terms.mk_int(BigInt::from(2));
        let four = terms.mk_int(BigInt::from(4));
        let ten = terms.mk_int(BigInt::from(10));

        let two_x = terms.mk_mul(vec![two, x]);
        let four_y = terms.mk_mul(vec![four, y]);

        let lhs = terms.mk_add(vec![two_x, four_y]);
        let eq = terms.mk_eq(lhs, ten);

        let mut solver = LiaSolver::new(&terms);
        solver.assert_literal(eq, true);

        let result = solver.check();
        // GCD test should NOT reject this (2 divides 10)
        // It may be SAT or NeedSplit (for branch-and-bound)
        assert!(
            !matches!(result, TheoryResult::Unsat(_)),
            "GCD test should not reject valid equation: {result:?}"
        );
    }

    #[test]
    fn test_lia_direct_enumeration_two_free_vars_fixes_solution() {
        let mut terms = TermStore::new();

        // Create vars in order so equalities pivot on (c, d) and leave (a, b) as free vars.
        let c = terms.mk_var("c", Sort::Int);
        let d = terms.mk_var("d", Sort::Int);
        let a = terms.mk_var("a", Sort::Int);
        let b = terms.mk_var("b", Sort::Int);

        let zero = terms.mk_int(BigInt::from(0));
        let one = terms.mk_int(BigInt::from(1));
        let minus_one = terms.mk_int(BigInt::from(-1));

        // c = a + b
        let a_plus_b = terms.mk_add(vec![a, b]);
        let eq_c = terms.mk_eq(c, a_plus_b);

        // d = a - b
        let neg_b = terms.mk_mul(vec![minus_one, b]);
        let a_minus_b = terms.mk_add(vec![a, neg_b]);
        let eq_d = terms.mk_eq(d, a_minus_b);

        // 0 <= a <= 1, 0 <= b <= 1
        let a_ge_0 = terms.mk_ge(a, zero);
        let a_le_1 = terms.mk_le(a, one);
        let b_ge_0 = terms.mk_ge(b, zero);
        let b_le_1 = terms.mk_le(b, one);

        let mut solver = LiaSolver::new(&terms);
        solver.assert_literal(eq_c, true);
        solver.assert_literal(eq_d, true);
        solver.assert_literal(a_ge_0, true);
        solver.assert_literal(a_le_1, true);
        solver.assert_literal(b_ge_0, true);
        solver.assert_literal(b_le_1, true);

        assert!(matches!(solver.lra.check(), TheoryResult::Sat));

        let (a_lb0, a_ub0) = solver.lra.get_bounds(a).unwrap();
        assert_ne!(a_lb0.as_ref().unwrap().value, a_ub0.as_ref().unwrap().value);

        let res = solver.try_direct_enumeration();
        assert!(res.is_none());

        let (a_lb1, a_ub1) = solver.lra.get_bounds(a).unwrap();
        assert_eq!(a_lb1.as_ref().unwrap().value, a_ub1.as_ref().unwrap().value);
        let (b_lb1, b_ub1) = solver.lra.get_bounds(b).unwrap();
        assert_eq!(b_lb1.as_ref().unwrap().value, b_ub1.as_ref().unwrap().value);
    }
}
