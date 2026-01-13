//! SMT backend for CHC solving
//!
//! This module provides SMT query support for the PDR solver by converting
//! CHC expressions to z4-core terms and using the z4-dpll solver.

use crate::{ChcExpr, ChcOp, ChcSort, ChcVar, PredicateId};
use num_bigint::BigInt;
use num_bigint::Sign;
use num_traits::{One, ToPrimitive};
use rustc_hash::FxHashMap;
use z4_core::term::{Constant, Symbol, TermData};
use z4_core::{Sort, TermId, TermStore};

/// Determine whether a term should be communicated to the theory solver.
/// Theory solvers only handle atomic predicates (like `x <= 5`, `x = y`),
/// not Boolean combinations (and/or/not/etc).
fn is_theory_atom(terms: &TermStore, term: TermId) -> bool {
    if terms.sort(term) != &Sort::Bool {
        return false;
    }

    match terms.get(term) {
        TermData::Const(Constant::Bool(_)) => false,
        TermData::Const(_) => false,
        TermData::Var(_, _) => false,
        TermData::Not(_) => false,
        TermData::Ite(_, _, _) => false,
        TermData::Let(_, _) => false,
        TermData::App(Symbol::Named(name), args) => match name.as_str() {
            "and" | "or" | "xor" | "=>" => false,
            "=" => {
                // Boolean equality is not a theory atom
                !(args.len() == 2
                    && terms.sort(args[0]) == &Sort::Bool
                    && terms.sort(args[1]) == &Sort::Bool)
            }
            _ => true,
        },
        TermData::App(_, _) => true,
    }
}

fn bigint_to_i64_saturating(v: &BigInt) -> i64 {
    if let Some(i) = v.to_i64() {
        return i;
    }
    match v.sign() {
        Sign::Minus => i64::MIN,
        Sign::NoSign | Sign::Plus => i64::MAX,
    }
}

/// UNSAT core - the subset of constraints that caused unsatisfiability
#[derive(Debug, Clone, Default)]
pub struct UnsatCore {
    /// Conjuncts from the original query that are sufficient for UNSAT.
    ///
    /// This is currently populated only for conjunction-shaped queries where
    /// we solve under assumptions and extract an UNSAT core over those assumptions.
    pub conjuncts: Vec<ChcExpr>,
}

/// Result of an SMT satisfiability check
#[derive(Debug, Clone)]
pub enum SmtResult {
    /// Formula is satisfiable, with a model mapping variable names to values
    Sat(FxHashMap<String, SmtValue>),
    /// Formula is unsatisfiable, optionally with an UNSAT core
    Unsat,
    /// Formula is unsatisfiable with an UNSAT core for interpolation
    UnsatWithCore(UnsatCore),
    /// Solver couldn't determine satisfiability
    Unknown,
}

/// A value in an SMT model
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtValue {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Bitvector value (value, width)
    BitVec(u64, u32),
}

/// SMT context for CHC solving
///
/// Converts CHC expressions to z4-core terms and provides satisfiability checking.
pub struct SmtContext {
    /// Term store for z4-core terms
    pub terms: TermStore,
    /// Mapping from CHC variable names to z4-core term IDs
    var_map: FxHashMap<String, TermId>,
    /// Mapping from predicate applications to boolean term IDs
    /// Key is (predicate_id, serialized args) for uniqueness
    pred_app_map: FxHashMap<(PredicateId, Vec<String>), TermId>,
    /// Counter for generating unique predicate application names
    pred_app_counter: u32,
    /// Optional wall-clock timeout for a single `check_sat` call.
    ///
    /// This is intended for best-effort, auxiliary queries (e.g. invariant discovery).
    check_timeout: Option<std::time::Duration>,
}

impl Default for SmtContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SmtContext {
    /// Create a new SMT context
    pub fn new() -> Self {
        SmtContext {
            terms: TermStore::new(),
            var_map: FxHashMap::default(),
            pred_app_map: FxHashMap::default(),
            pred_app_counter: 0,
            check_timeout: None,
        }
    }

    /// Reset the context
    pub fn reset(&mut self) {
        self.terms = TermStore::new();
        self.var_map.clear();
        self.pred_app_map.clear();
        self.pred_app_counter = 0;
        self.check_timeout = None;
    }

    /// Run a satisfiability check with a wall-clock timeout.
    ///
    /// On timeout, returns `SmtResult::Unknown`.
    pub fn check_sat_with_timeout(
        &mut self,
        expr: &ChcExpr,
        timeout: std::time::Duration,
    ) -> SmtResult {
        let prev = self.check_timeout.replace(timeout);
        let result = self.check_sat(expr);
        self.check_timeout = prev;
        result
    }

    /// Convert a CHC sort to a z4-core sort
    pub fn convert_sort(sort: &ChcSort) -> Sort {
        match sort {
            ChcSort::Bool => Sort::Bool,
            ChcSort::Int => Sort::Int,
            ChcSort::Real => Sort::Real,
            ChcSort::BitVec(w) => Sort::BitVec(*w),
            ChcSort::Array(key, val) => Sort::Array(
                Box::new(Self::convert_sort(key)),
                Box::new(Self::convert_sort(val)),
            ),
        }
    }

    /// Get or create a term for a CHC variable
    fn get_or_create_var(&mut self, var: &ChcVar) -> TermId {
        if let Some(&term) = self.var_map.get(&var.name) {
            return term;
        }

        let sort = Self::convert_sort(&var.sort);
        let term = self.terms.mk_var(&var.name, sort);
        self.var_map.insert(var.name.clone(), term);
        term
    }

    /// Convert a CHC expression to a z4-core term
    pub fn convert_expr(&mut self, expr: &ChcExpr) -> TermId {
        match expr {
            ChcExpr::Bool(b) => self.terms.mk_bool(*b),

            ChcExpr::Int(n) => self.terms.mk_int(BigInt::from(*n)),

            ChcExpr::Real(num, denom) => {
                use num_rational::BigRational;
                let r = BigRational::new(BigInt::from(*num), BigInt::from(*denom));
                self.terms.mk_rational(r)
            }

            ChcExpr::Var(v) => self.get_or_create_var(v),

            ChcExpr::PredicateApp(name, id, args) => {
                // Serialize arguments for uniqueness key
                let arg_strs: Vec<String> = args.iter().map(|a| format!("{}", a)).collect();
                let key = (*id, arg_strs);

                if let Some(&term) = self.pred_app_map.get(&key) {
                    return term;
                }

                // Create a fresh boolean variable for this predicate application
                let var_name = format!("{}_{}", name, self.pred_app_counter);
                self.pred_app_counter += 1;
                let term = self.terms.mk_var(&var_name, Sort::Bool);
                self.pred_app_map.insert(key, term);
                term
            }

            ChcExpr::Op(op, args) => {
                let term_args: Vec<TermId> = args.iter().map(|a| self.convert_expr(a)).collect();

                match op {
                    ChcOp::Not => {
                        assert_eq!(term_args.len(), 1);
                        self.terms.mk_not(term_args[0])
                    }
                    ChcOp::And => self.terms.mk_and(term_args),
                    ChcOp::Or => self.terms.mk_or(term_args),
                    ChcOp::Implies => {
                        assert_eq!(term_args.len(), 2);
                        self.terms.mk_implies(term_args[0], term_args[1])
                    }
                    ChcOp::Iff => {
                        assert_eq!(term_args.len(), 2);
                        // a <-> b is (a => b) /\ (b => a)
                        let ab = self.terms.mk_implies(term_args[0], term_args[1]);
                        let ba = self.terms.mk_implies(term_args[1], term_args[0]);
                        self.terms.mk_and(vec![ab, ba])
                    }
                    ChcOp::Add => self.terms.mk_add(term_args),
                    ChcOp::Sub => self.terms.mk_sub(term_args),
                    ChcOp::Mul => self.terms.mk_mul(term_args),
                    ChcOp::Div => self.terms.mk_intdiv(term_args),
                    ChcOp::Mod => self.terms.mk_mod(term_args),
                    ChcOp::Neg => {
                        assert_eq!(term_args.len(), 1);
                        self.terms.mk_neg(term_args[0])
                    }
                    ChcOp::Eq => {
                        assert_eq!(term_args.len(), 2);
                        self.terms.mk_eq(term_args[0], term_args[1])
                    }
                    ChcOp::Ne => {
                        assert_eq!(term_args.len(), 2);
                        // Encode `a != b` as `not (a = b)` rather than `distinct(a, b)`.
                        //
                        // `distinct` is a theory atom and requires explicit disequality support in
                        // theory solvers. Encoding as `not (= ...)` allows DPLL(T) to treat it as a
                        // Boolean negation of an equality atom, which is more robust for Z4's CHC
                        // auxiliary queries (e.g., invariant preservation checks).
                        let eq = self.terms.mk_eq(term_args[0], term_args[1]);
                        self.terms.mk_not(eq)
                    }
                    ChcOp::Lt => {
                        assert_eq!(term_args.len(), 2);
                        self.terms.mk_lt(term_args[0], term_args[1])
                    }
                    ChcOp::Le => {
                        assert_eq!(term_args.len(), 2);
                        self.terms.mk_le(term_args[0], term_args[1])
                    }
                    ChcOp::Gt => {
                        assert_eq!(term_args.len(), 2);
                        self.terms.mk_gt(term_args[0], term_args[1])
                    }
                    ChcOp::Ge => {
                        assert_eq!(term_args.len(), 2);
                        self.terms.mk_ge(term_args[0], term_args[1])
                    }
                    ChcOp::Ite => {
                        assert_eq!(term_args.len(), 3);
                        self.terms.mk_ite(term_args[0], term_args[1], term_args[2])
                    }
                    ChcOp::Select => {
                        assert_eq!(term_args.len(), 2);
                        self.terms.mk_select(term_args[0], term_args[1])
                    }
                    ChcOp::Store => {
                        assert_eq!(term_args.len(), 3);
                        self.terms
                            .mk_store(term_args[0], term_args[1], term_args[2])
                    }
                }
            }
        }
    }

    fn flatten_top_level_and(expr: &ChcExpr, out: &mut Vec<ChcExpr>) {
        match expr {
            ChcExpr::Op(ChcOp::And, args) => {
                for a in args {
                    Self::flatten_top_level_and(a, out);
                }
            }
            _ => out.push(expr.clone()),
        }
    }

    fn collect_int_var_const_equalities(expr: &ChcExpr, out: &mut FxHashMap<String, i64>) {
        match expr {
            ChcExpr::Op(ChcOp::And, args) => {
                for a in args {
                    Self::collect_int_var_const_equalities(a, out);
                }
            }
            ChcExpr::Op(ChcOp::Eq, args) if args.len() == 2 => {
                match (args[0].as_ref(), args[1].as_ref()) {
                    (ChcExpr::Var(v), ChcExpr::Int(n)) if matches!(v.sort, ChcSort::Int) => {
                        out.insert(v.name.clone(), *n);
                    }
                    (ChcExpr::Int(n), ChcExpr::Var(v)) if matches!(v.sort, ChcSort::Int) => {
                        out.insert(v.name.clone(), *n);
                    }
                    _ => {
                        if let Some((v, val)) =
                            Self::extract_linear_int_equality(args[0].as_ref(), args[1].as_ref())
                                .or_else(|| {
                                    Self::extract_linear_int_equality(
                                        args[1].as_ref(),
                                        args[0].as_ref(),
                                    )
                                })
                        {
                            out.insert(v.name.clone(), val);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn extract_linear_int_equality(lhs: &ChcExpr, rhs: &ChcExpr) -> Option<(ChcVar, i64)> {
        let c = match rhs {
            ChcExpr::Int(n) => *n,
            _ => return None,
        };

        match lhs {
            // (+ k var) = c => var = c - k
            ChcExpr::Op(ChcOp::Add, inner) if inner.len() == 2 => {
                match (inner[0].as_ref(), inner[1].as_ref()) {
                    (ChcExpr::Int(k), ChcExpr::Var(v)) | (ChcExpr::Var(v), ChcExpr::Int(k))
                        if matches!(v.sort, ChcSort::Int) =>
                    {
                        Some((v.clone(), c - k))
                    }
                    _ => None,
                }
            }
            // (* -1 var) = c => var = -c
            ChcExpr::Op(ChcOp::Mul, inner) if inner.len() == 2 => {
                match (inner[0].as_ref(), inner[1].as_ref()) {
                    (ChcExpr::Int(-1), ChcExpr::Var(v)) | (ChcExpr::Var(v), ChcExpr::Int(-1))
                        if matches!(v.sort, ChcSort::Int) =>
                    {
                        Some((v.clone(), -c))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn collect_int_var_bounds(
        expr: &ChcExpr,
        lower: &mut FxHashMap<String, i64>,
        upper: &mut FxHashMap<String, i64>,
    ) {
        match expr {
            ChcExpr::Op(ChcOp::And, args) => {
                for a in args {
                    Self::collect_int_var_bounds(a, lower, upper);
                }
            }
            ChcExpr::Op(ChcOp::Le | ChcOp::Ge, args) if args.len() == 2 => {
                let op = match expr {
                    ChcExpr::Op(op, _) => op,
                    _ => unreachable!(),
                };

                let (a, b) = (args[0].as_ref(), args[1].as_ref());

                let update_lower = |name: &str, v: i64, lower: &mut FxHashMap<String, i64>| {
                    lower
                        .entry(name.to_string())
                        .and_modify(|cur| *cur = (*cur).max(v))
                        .or_insert(v);
                };
                let update_upper = |name: &str, v: i64, upper: &mut FxHashMap<String, i64>| {
                    upper
                        .entry(name.to_string())
                        .and_modify(|cur| *cur = (*cur).min(v))
                        .or_insert(v);
                };

                match (op, a, b) {
                    // var <= c
                    (ChcOp::Le, ChcExpr::Var(v), ChcExpr::Int(c))
                        if matches!(v.sort, ChcSort::Int) =>
                    {
                        update_upper(&v.name, *c, upper);
                    }
                    // var >= c
                    (ChcOp::Ge, ChcExpr::Var(v), ChcExpr::Int(c))
                        if matches!(v.sort, ChcSort::Int) =>
                    {
                        update_lower(&v.name, *c, lower);
                    }
                    // c <= var  => var >= c
                    (ChcOp::Le, ChcExpr::Int(c), ChcExpr::Var(v))
                        if matches!(v.sort, ChcSort::Int) =>
                    {
                        update_lower(&v.name, *c, lower);
                    }
                    // c >= var  => var <= c
                    (ChcOp::Ge, ChcExpr::Int(c), ChcExpr::Var(v))
                        if matches!(v.sort, ChcSort::Int) =>
                    {
                        update_upper(&v.name, *c, upper);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn term_const_i64_if_int(&self, term: TermId) -> Option<i64> {
        match self.terms.get(term) {
            TermData::Const(Constant::Int(n)) => Some(bigint_to_i64_saturating(n)),
            TermData::Const(Constant::Rational(r)) => {
                if r.0.denom() == &BigInt::from(1) {
                    Some(bigint_to_i64_saturating(r.0.numer()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Infer a conservative inclusive integer interval `[lb, ub]` for `variable` from the current
    /// SAT assignment over atomic comparisons.
    ///
    /// This is used to prune infeasible branches for disequality splits (common for `mod` remainders
    /// where bounds like `0 <= r < 2` make `r != 0` deterministically imply `r > 0`).
    fn infer_int_bounds_from_sat_model<'a, I>(
        &self,
        variable: TermId,
        model: &[bool],
        var_to_term: I,
    ) -> (Option<i64>, Option<i64>)
    where
        I: IntoIterator<Item = (&'a u32, &'a TermId)>,
    {
        if self.terms.sort(variable) != &Sort::Int {
            return (None, None);
        }

        let mut lower: Option<i64> = None;
        let mut upper: Option<i64> = None;

        for (&var_idx, &term_id) in var_to_term {
            let sat_var = z4_sat::Variable(var_idx - 1);
            let Some(&assigned) = model.get(sat_var.0 as usize) else {
                continue;
            };

            let TermData::App(sym, args) = self.terms.get(term_id) else {
                continue;
            };
            if args.len() != 2 {
                continue;
            }

            let op = sym.name();

            // We only extract bounds from atoms where one side is exactly `variable`
            // and the other side is an (integer) constant.
            let (var_on_left, c) = if args[0] == variable {
                let Some(c) = self.term_const_i64_if_int(args[1]) else {
                    continue;
                };
                (true, c)
            } else if args[1] == variable {
                let Some(c) = self.term_const_i64_if_int(args[0]) else {
                    continue;
                };
                (false, c)
            } else {
                continue;
            };

            // Convert atom + assignment into an implied inclusive bound on `variable`.
            // Handle `variable < c` as `variable <= c-1` for Int, etc.
            let mut apply_lower = |v: i64| {
                lower = Some(lower.map_or(v, |cur| cur.max(v)));
            };
            let mut apply_upper = |v: i64| {
                upper = Some(upper.map_or(v, |cur| cur.min(v)));
            };

            match (op, var_on_left, assigned) {
                // variable < c
                ("<", true, true) => {
                    if let Some(v) = c.checked_sub(1) {
                        apply_upper(v);
                    }
                }
                ("<", true, false) => apply_lower(c),
                // c < variable
                ("<", false, true) => {
                    if let Some(v) = c.checked_add(1) {
                        apply_lower(v);
                    }
                }
                ("<", false, false) => apply_upper(c),

                // variable <= c
                ("<=", true, true) => apply_upper(c),
                ("<=", true, false) => {
                    if let Some(v) = c.checked_add(1) {
                        apply_lower(v);
                    }
                }
                // c <= variable
                ("<=", false, true) => apply_lower(c),
                ("<=", false, false) => {
                    if let Some(v) = c.checked_sub(1) {
                        apply_upper(v);
                    }
                }

                // variable > c
                (">", true, true) => {
                    if let Some(v) = c.checked_add(1) {
                        apply_lower(v);
                    }
                }
                (">", true, false) => apply_upper(c),
                // c > variable
                (">", false, true) => {
                    if let Some(v) = c.checked_sub(1) {
                        apply_upper(v);
                    }
                }
                (">", false, false) => apply_lower(c),

                // variable >= c
                (">=", true, true) => apply_lower(c),
                (">=", true, false) => {
                    if let Some(v) = c.checked_sub(1) {
                        apply_upper(v);
                    }
                }
                // c >= variable
                (">=", false, true) => apply_upper(c),
                (">=", false, false) => {
                    if let Some(v) = c.checked_add(1) {
                        apply_lower(v);
                    }
                }

                // variable = c (only use when true)
                ("=", true, true) => {
                    apply_lower(c);
                    apply_upper(c);
                }
                ("=", false, true) => {
                    apply_lower(c);
                    apply_upper(c);
                }

                _ => {}
            }
        }

        (lower, upper)
    }

    fn const_as_big_rational(&self, term: TermId) -> Option<num_rational::BigRational> {
        match self.terms.get(term) {
            TermData::Const(Constant::Int(n)) => Some(num_rational::BigRational::from(n.clone())),
            TermData::Const(Constant::Rational(r)) => Some(r.0.clone()),
            _ => None,
        }
    }

    fn find_diseq_guard_atom<'a, I>(
        &self,
        split_variable: TermId,
        excluded_value: &num_rational::BigRational,
        model: &[bool],
        var_to_term: I,
    ) -> Option<(TermId, DiseqGuardKind)>
    where
        I: IntoIterator<Item = (&'a u32, &'a TermId)>,
    {
        for (&var_idx, &term_id) in var_to_term {
            let sat_var = z4_sat::Variable(var_idx - 1);
            let Some(&assigned) = model.get(sat_var.0 as usize) else {
                continue;
            };

            let TermData::App(sym, args) = self.terms.get(term_id) else {
                continue;
            };
            if args.len() != 2 {
                continue;
            }

            let sym_name = sym.name();
            if sym_name != "=" && sym_name != "distinct" {
                continue;
            }

            let matches = |a: TermId, b: TermId| -> bool {
                if a != split_variable {
                    return false;
                }
                let Some(c) = self.const_as_big_rational(b) else {
                    return false;
                };
                &c == excluded_value
            };

            let arg0 = args[0];
            let arg1 = args[1];
            let is_match = matches(arg0, arg1) || matches(arg1, arg0);
            if !is_match {
                continue;
            }

            // Only treat this atom as the guard if it currently asserts the disequality:
            // - (distinct x c) asserted true
            // - (= x c) asserted false
            if sym_name == "distinct" && assigned {
                return Some((term_id, DiseqGuardKind::Distinct));
            }
            if sym_name == "=" && !assigned {
                return Some((term_id, DiseqGuardKind::Eq));
            }
        }
        None
    }

    /// Check satisfiability of a CHC expression using LIA theory
    pub fn check_sat(&mut self, expr: &ChcExpr) -> SmtResult {
        use z4_core::term::TermData;
        use z4_core::{TheoryResult, TheorySolver, Tseitin};
        use z4_lia::LiaSolver;

        let start = std::time::Instant::now();
        let timeout = self.check_timeout;

        // Step 1: Apply constant propagation to enable folding of mod expressions
        // e.g., (A = 0) ∧ (mod A 2) != 0 becomes (A = 0) ∧ (0 != 0) = false
        //
        // We also preserve `var = const` bindings from the original formula so PDR can still
        // recover them even if propagation simplifies the equalities to `true`.
        let mut propagated_equalities: FxHashMap<String, i64> = FxHashMap::default();
        Self::collect_int_var_const_equalities(expr, &mut propagated_equalities);
        let simplified = expr.propagate_constants();

        // Propagation can discover new bindings (e.g. `(+ 1 A) = 4` ⇒ `A = 3`) and also
        // eliminate the original equality conjuncts. Preserve any bindings visible after
        // propagation so PDR can still reconstruct predecessor states.
        Self::collect_int_var_const_equalities(&simplified, &mut propagated_equalities);

        let mut propagated_model: FxHashMap<String, SmtValue> = FxHashMap::default();
        for (name, value) in &propagated_equalities {
            propagated_model.insert(name.clone(), SmtValue::Int(*value));
        }

        // Check if simplified to a constant - this is the main benefit of propagation
        if let ChcExpr::Bool(b) = &simplified {
            return if *b {
                SmtResult::Sat(propagated_model)
            } else {
                SmtResult::Unsat
            };
        }

        // Step 2: Eliminate arithmetic ite expressions, which are not supported by the LIA solver.
        let ite_eliminated = simplified.eliminate_ite();

        // Step 3: Eliminate mod/div expressions by introducing auxiliary variables.
        // This allows the LIA solver to handle integer division/modulo when the divisor is constant.
        let mod_eliminated = ite_eliminated.eliminate_mod();

        // Step 4: Normalize negations so theory atoms are visible to the LIA solver.
        let normalized = mod_eliminated.normalize_negations();
        // Step 5: Rewrite strict integer bounds to non-strict equivalents.
        let mut normalized = normalized.normalize_strict_int_comparisons();

        // Step 6: Promote singleton bounds to equalities (e.g. A>=0 ∧ A<=0 ⇒ A=0) and
        // propagate them. This substantially simplifies disjunction-heavy formulas produced
        // by point-exclusion lemmas and prevents the backend from getting stuck on repeated
        // disequality splitting.
        let mut lower_bounds: FxHashMap<String, i64> = FxHashMap::default();
        let mut upper_bounds: FxHashMap<String, i64> = FxHashMap::default();
        Self::collect_int_var_bounds(&normalized, &mut lower_bounds, &mut upper_bounds);

        let mut bound_subst: Vec<(ChcVar, ChcExpr)> = Vec::new();
        for (name, lb) in &lower_bounds {
            let Some(ub) = upper_bounds.get(name) else {
                continue;
            };
            if lb == ub {
                propagated_equalities.entry(name.clone()).or_insert(*lb);
                bound_subst.push((ChcVar::new(name, ChcSort::Int), ChcExpr::Int(*lb)));
            }
        }
        if !bound_subst.is_empty() {
            normalized = normalized.substitute(&bound_subst).simplify_constants();
            propagated_model.clear();
            for (name, value) in &propagated_equalities {
                propagated_model.insert(name.clone(), SmtValue::Int(*value));
            }
        }

        // Try assumption-based solving for conjunction-shaped queries.
        // This enables UNSAT core extraction over top-level conjuncts.
        let mut top_conjuncts = Vec::new();
        Self::flatten_top_level_and(&normalized, &mut top_conjuncts);

        // Trivial conjunctions.
        if top_conjuncts.is_empty() {
            return SmtResult::Sat(propagated_model);
        }
        if top_conjuncts
            .iter()
            .any(|c| matches!(c, ChcExpr::Bool(false)))
        {
            return SmtResult::UnsatWithCore(UnsatCore {
                conjuncts: vec![ChcExpr::Bool(false)],
            });
        }
        top_conjuncts.retain(|c| !matches!(c, ChcExpr::Bool(true)));
        if top_conjuncts.is_empty() {
            return SmtResult::Sat(propagated_model);
        }

        // Use assumption solving only when we actually have multiple conjuncts
        // (otherwise the core is uninformative).
        let use_assumptions = top_conjuncts.len() >= 2;

        // Build CNF via Tseitin.
        let (mut term_to_var, mut var_to_term, mut num_vars, mut sat, assumptions, assumption_map) =
            if use_assumptions {
                let conjunct_terms: Vec<(ChcExpr, TermId)> = top_conjuncts
                    .iter()
                    .map(|c| (c.clone(), self.convert_expr(c)))
                    .collect();

                let mut tseitin = Tseitin::new(&self.terms);

                let mut assumptions: Vec<z4_sat::Literal> =
                    Vec::with_capacity(conjunct_terms.len());
                let mut assumption_map: FxHashMap<z4_sat::Literal, ChcExpr> = FxHashMap::default();

                for (c, c_term) in &conjunct_terms {
                    let cnf_lit = tseitin.encode(*c_term, true);
                    let sat_lit = cnf_lit_to_sat_lit(cnf_lit);
                    assumptions.push(sat_lit);
                    assumption_map.insert(sat_lit, c.clone());
                }

                let mut sat = z4_sat::Solver::new(tseitin.num_vars() as usize);
                for clause in tseitin.all_clauses() {
                    let lits: Vec<z4_sat::Literal> = clause
                        .0
                        .iter()
                        .map(|&lit| cnf_lit_to_sat_lit(lit))
                        .collect();
                    sat.add_clause(lits);
                }

                (
                    tseitin.term_to_var().clone(),
                    tseitin.var_to_term().clone(),
                    tseitin.num_vars(),
                    sat,
                    Some(assumptions),
                    Some(assumption_map),
                )
            } else {
                // Fall back to the legacy "assert root" encoding for non-conjunction queries.
                let term = self.convert_expr(&normalized);
                let tseitin = Tseitin::new(&self.terms);
                let result = tseitin.transform(term);

                let mut sat = z4_sat::Solver::new(result.num_vars as usize);
                for clause in &result.clauses {
                    let lits: Vec<z4_sat::Literal> = clause
                        .0
                        .iter()
                        .map(|&lit| cnf_lit_to_sat_lit(lit))
                        .collect();
                    sat.add_clause(lits);
                }

                (
                    result.term_to_var.clone(),
                    result.var_to_term.clone(),
                    result.num_vars,
                    sat,
                    None,
                    None,
                )
            };

        // Track number of splits to prevent infinite loops
        let mut split_count: usize = 0;
        const MAX_SPLITS: usize = 10000;

        // Track per-variable disequality splits to detect infinite enumeration
        // Key: variable TermId, Value: count of splits on that variable
        let mut var_diseq_splits: FxHashMap<TermId, usize> = FxHashMap::default();
        const MAX_VAR_SPLITS: usize = 200; // Baseline per-variable split cap (unbounded case)

        // Solve (DPLL(T) loop). Bound iterations to avoid pathological non-termination.
        let mut dt_iterations: usize = 0;
        let debug = std::env::var("Z4_DEBUG_CHC_SMT").is_ok();
        loop {
            if let Some(timeout) = timeout {
                if start.elapsed() >= timeout {
                    if debug {
                        eprintln!("[CHC-SMT] Timeout exceeded");
                    }
                    return SmtResult::Unknown;
                }
            }
            dt_iterations += 1;
            if dt_iterations > 10_000 {
                if debug {
                    eprintln!("[CHC-SMT] Exceeded max iterations");
                }
                return SmtResult::Unknown;
            }
            let sat_result = if let Some(assumptions) = &assumptions {
                sat.solve_with_assumptions(assumptions)
            } else {
                match sat.solve() {
                    z4_sat::SolveResult::Sat(model) => z4_sat::AssumeResult::Sat(model),
                    z4_sat::SolveResult::Unsat => z4_sat::AssumeResult::Unsat(Vec::new()),
                    z4_sat::SolveResult::Unknown => z4_sat::AssumeResult::Unknown,
                }
            };

            match sat_result {
                z4_sat::AssumeResult::Sat(model) => {
                    // Create LIA solver fresh each iteration (allows term store mutations)
                    let mut lia = LiaSolver::new(&self.terms);

                    // Sync theory solver with atomic constraints only (not Boolean combinations)
                    for (&var_idx, &term_id) in &var_to_term {
                        // Skip Boolean combinations - theory solver only handles atomic predicates
                        if !is_theory_atom(&self.terms, term_id) {
                            if debug {
                                eprintln!(
                                    "[CHC-SMT]   SKIPPED non-theory-atom {:?} = {:?}",
                                    term_id,
                                    self.terms.get(term_id)
                                );
                            }
                            continue;
                        }
                        // CNF vars are 1-indexed, SAT model is 0-indexed
                        let sat_var = z4_sat::Variable(var_idx - 1);
                        if let Some(value) = model.get(sat_var.0 as usize) {
                            lia.assert_literal(term_id, *value);
                        }
                    }

                    if debug {
                        eprintln!(
                            "[CHC-SMT] iter {}: SAT model, asserting {} terms to LIA",
                            dt_iterations,
                            var_to_term.len()
                        );
                        for (&var_idx, &term_id) in &var_to_term {
                            let sat_var = z4_sat::Variable(var_idx - 1);
                            if let Some(value) = model.get(sat_var.0 as usize) {
                                eprintln!("[CHC-SMT]   term {:?} = {}", term_id, value);
                            }
                        }
                    }
                    let lia_result = lia.check();
                    if debug {
                        eprintln!(
                            "[CHC-SMT] iter {}: LIA result: {:?}",
                            dt_iterations, lia_result
                        );
                    }
                    match lia_result {
                        TheoryResult::Sat => {
                            // Build model from SAT assignment
                            let mut values = FxHashMap::default();
                            let lia_model = lia.extract_model();

                            for (name, &term_id) in &self.var_map {
                                match self.terms.sort(term_id) {
                                    Sort::Bool => {
                                        if let Some(&cnf_var) = term_to_var.get(&term_id) {
                                            let sat_var = z4_sat::Variable(cnf_var - 1);
                                            if let Some(value) = model.get(sat_var.0 as usize) {
                                                values.insert(name.clone(), SmtValue::Bool(*value));
                                            }
                                        }
                                    }
                                    Sort::Int => {
                                        if let Some(m) = &lia_model {
                                            if let Some(v) = m.values.get(&term_id) {
                                                values.insert(
                                                    name.clone(),
                                                    SmtValue::Int(bigint_to_i64_saturating(v)),
                                                );
                                                continue;
                                            }
                                        }

                                        // Fallback: LIA may not include all `Int` vars in its extracted model,
                                        // but the underlying LRA solver still tracks their values.
                                        if let Some(v) = lia.lra_solver().get_value(term_id) {
                                            if v.denom().is_one() {
                                                values.insert(
                                                    name.clone(),
                                                    SmtValue::Int(bigint_to_i64_saturating(
                                                        v.numer(),
                                                    )),
                                                );
                                            } else {
                                                return SmtResult::Unknown;
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            // Check equality/disequality/comparison constraints that LIA might have missed
                            // The LIA solver handles linear arithmetic but may not catch all constraint violations
                            let mut violated_constraint = None;
                            for (&var_idx, &term_id) in &var_to_term {
                                let sat_var = z4_sat::Variable(var_idx - 1);
                                let Some(&sat_value) = model.get(sat_var.0 as usize) else {
                                    continue;
                                };

                                // Check if this term is a comparison or equality
                                if let TermData::App(sym, args) = self.terms.get(term_id) {
                                    let sym_name = sym.name();

                                    if args.len() == 2 {
                                        // Get the values of both sides
                                        let lhs = args[0];
                                        let rhs = args[1];

                                        let lhs_val = self.get_term_value(lhs, &values, &lia_model);
                                        let rhs_val = self.get_term_value(rhs, &values, &lia_model);

                                        if let (Some(l), Some(r)) = (lhs_val, rhs_val) {
                                            // Compute the actual truth value of the constraint
                                            let actual_value = match sym_name {
                                                "=" => l == r,
                                                "distinct" => l != r,
                                                "<" => l < r,
                                                "<=" => l <= r,
                                                ">" => l > r,
                                                ">=" => l >= r,
                                                _ => continue, // Unknown operator, skip
                                            };

                                            // Check if the SAT assignment is consistent with the actual value
                                            if sat_value != actual_value {
                                                violated_constraint = Some((var_idx, sat_value));
                                                break;
                                            }
                                        }
                                    }
                                }
                            }

                            if let Some((violated_var, violated_val)) = violated_constraint {
                                // Add blocking clause to exclude this assignment
                                let sat_var = z4_sat::Variable(violated_var - 1);
                                let blocking_lit = if violated_val {
                                    z4_sat::Literal::negative(sat_var)
                                } else {
                                    z4_sat::Literal::positive(sat_var)
                                };
                                sat.add_clause(vec![blocking_lit]);
                                continue; // Try again with the blocking clause
                            }

                            for (name, value) in &propagated_model {
                                values.entry(name.clone()).or_insert_with(|| value.clone());
                            }
                            return SmtResult::Sat(values);
                        }
                        TheoryResult::Unsat(conflict) => {
                            // Add blocking clause
                            if conflict.is_empty() {
                                return SmtResult::Unsat;
                            }
                            let clause: Vec<z4_sat::Literal> = conflict
                                .iter()
                                .filter_map(|lit| {
                                    term_to_var.get(&lit.term).map(|&cnf_var| {
                                        let sat_var = z4_sat::Variable(cnf_var - 1);
                                        if lit.value {
                                            z4_sat::Literal::negative(sat_var)
                                        } else {
                                            z4_sat::Literal::positive(sat_var)
                                        }
                                    })
                                })
                                .collect();
                            if clause.is_empty() {
                                return SmtResult::Unsat;
                            }
                            sat.add_clause(clause);
                        }
                        TheoryResult::Unknown => {
                            return SmtResult::Unknown;
                        }
                        TheoryResult::NeedSplit(split) => {
                            // LIA needs branch-and-bound: the LRA relaxation found x = frac,
                            // so we must explore (x <= floor) OR (x >= ceil)

                            split_count += 1;
                            if split_count > MAX_SPLITS {
                                if debug {
                                    eprintln!("[CHC-SMT] Exceeded max splits ({})", MAX_SPLITS);
                                }
                                return SmtResult::Unknown;
                            }

                            if debug {
                                eprintln!("[CHC-SMT] iter {}: NeedSplit on var {:?}, value={}, floor={}, ceil={}",
                                    dt_iterations, split.variable, split.value, split.floor, split.ceil);
                            }

                            // Create split atoms: (var <= floor) and (var >= ceil)
                            let floor_term = self.terms.mk_int(split.floor.clone());
                            let ceil_term = self.terms.mk_int(split.ceil.clone());
                            let le_atom = self.terms.mk_le(split.variable, floor_term);
                            let ge_atom = self.terms.mk_ge(split.variable, ceil_term);

                            // Allocate CNF variables for the split atoms if not already assigned
                            let le_var = *term_to_var.entry(le_atom).or_insert_with(|| {
                                num_vars += 1;
                                var_to_term.insert(num_vars, le_atom);
                                num_vars
                            });
                            let ge_var = *term_to_var.entry(ge_atom).or_insert_with(|| {
                                num_vars += 1;
                                var_to_term.insert(num_vars, ge_atom);
                                num_vars
                            });

                            // IMPORTANT: Tell the SAT solver about any new variables before adding clauses
                            sat.ensure_num_vars(num_vars as usize);

                            // Add splitting clause: (le_atom OR ge_atom)
                            // This enforces that at least one branch must be taken
                            let le_sat_var = z4_sat::Variable(le_var - 1);
                            let ge_sat_var = z4_sat::Variable(ge_var - 1);
                            let split_clause = vec![
                                z4_sat::Literal::positive(le_sat_var),
                                z4_sat::Literal::positive(ge_sat_var),
                            ];
                            sat.add_clause(split_clause);

                            if debug {
                                eprintln!(
                                    "[CHC-SMT] Added split clause: le_var={}, ge_var={}",
                                    le_var, ge_var
                                );
                            }

                            // Continue the DPLL(T) loop - SAT solver will now explore both branches
                            continue;
                        }
                        TheoryResult::NeedDisequlitySplit(split) => {
                            // Disequality split: var != excluded_value
                            // Create atoms (var < excluded_value) OR (var > excluded_value)

                            split_count += 1;
                            if split_count > MAX_SPLITS {
                                if debug {
                                    eprintln!("[CHC-SMT] Exceeded max splits ({})", MAX_SPLITS);
                                }
                                return SmtResult::Unknown;
                            }

                            // Track per-variable splits to detect infinite enumeration
                            let (lb, ub) = if self.terms.sort(split.variable) == &Sort::Int
                                && split.excluded_value.is_integer()
                            {
                                self.infer_int_bounds_from_sat_model(
                                    split.variable,
                                    &model,
                                    var_to_term.iter(),
                                )
                            } else {
                                (None, None)
                            };
                            let max_var_splits = match (lb, ub) {
                                (Some(l), Some(u)) if l <= u => {
                                    let range_size = u.saturating_sub(l).saturating_add(1);
                                    if (1..=256).contains(&range_size) {
                                        MAX_VAR_SPLITS.max(range_size as usize + 2)
                                    } else {
                                        MAX_VAR_SPLITS
                                    }
                                }
                                _ => MAX_VAR_SPLITS,
                            };

                            let var_count = var_diseq_splits.entry(split.variable).or_insert(0);
                            *var_count += 1;
                            if *var_count > max_var_splits {
                                if debug {
                                    eprintln!(
                                        "[CHC-SMT] Exceeded max per-variable splits ({}) for {:?} - returning Unknown",
                                        max_var_splits, split.variable
                                    );
                                }
                                return SmtResult::Unknown;
                            }

                            if debug {
                                eprintln!("[CHC-SMT] iter {}: NeedDisequlitySplit on var {:?}, excluded={} (count={})",
                                    dt_iterations, split.variable, split.excluded_value, *var_count);
                            }

                            // If we can infer tight integer bounds from the current SAT assignment,
                            // prune infeasible branches to avoid pathological split blowups (common
                            // for `mod` remainder vars with bounds like `0 <= r < 2`).
                            let (can_lt, can_gt) = if split.excluded_value.is_integer() {
                                let excluded_i64 =
                                    bigint_to_i64_saturating(split.excluded_value.numer());
                                let lt_threshold = excluded_i64.checked_sub(1);
                                let gt_threshold = excluded_i64.checked_add(1);

                                let can_lt =
                                    lt_threshold.is_some_and(|t| lb.is_none_or(|l| l <= t));
                                let can_gt =
                                    gt_threshold.is_some_and(|t| ub.is_none_or(|u| u >= t));
                                (can_lt, can_gt)
                            } else {
                                (true, true)
                            };

                            // For integer variables with integer excluded values, avoid strict
                            // inequalities here. Strict bounds are harder to handle robustly in
                            // the LRA/LIA backend and can allow `x = excluded` to slip through.
                            // Encode `x != k` as `(x <= k-1) OR (x >= k+1)`.
                            let (left_atom, right_atom) = if self.terms.sort(split.variable)
                                == &Sort::Int
                                && split.excluded_value.is_integer()
                            {
                                let excluded = split.excluded_value.numer().clone();
                                let le_bound = self.terms.mk_int(&excluded - BigInt::from(1));
                                let ge_bound = self.terms.mk_int(&excluded + BigInt::from(1));
                                let left = self.terms.mk_le(split.variable, le_bound);
                                let right = self.terms.mk_ge(split.variable, ge_bound);
                                if debug {
                                    eprintln!(
                                            "[CHC-SMT] Created split atoms: left={:?} ({:?}), right={:?} ({:?})",
                                            left, self.terms.get(left), right, self.terms.get(right)
                                        );
                                }
                                (left, right)
                            } else {
                                let excluded_term =
                                    self.terms.mk_rational(split.excluded_value.clone());
                                (
                                    self.terms.mk_lt(split.variable, excluded_term),
                                    self.terms.mk_gt(split.variable, excluded_term),
                                )
                            };

                            // Allocate CNF variables for the split atoms
                            let lt_var = *term_to_var.entry(left_atom).or_insert_with(|| {
                                num_vars += 1;
                                var_to_term.insert(num_vars, left_atom);
                                num_vars
                            });
                            let gt_var = *term_to_var.entry(right_atom).or_insert_with(|| {
                                num_vars += 1;
                                var_to_term.insert(num_vars, right_atom);
                                num_vars
                            });

                            // Tell the SAT solver about new variables
                            sat.ensure_num_vars(num_vars as usize);

                            let lt_sat_var = z4_sat::Variable(lt_var - 1);
                            let gt_sat_var = z4_sat::Variable(gt_var - 1);
                            let guard_lit = self
                                .find_diseq_guard_atom(
                                    split.variable,
                                    &split.excluded_value,
                                    &model,
                                    var_to_term.iter(),
                                )
                                .and_then(|(guard_term, kind)| {
                                    term_to_var.get(&guard_term).copied().map(|cnf_var| {
                                        let sat_var = z4_sat::Variable(cnf_var - 1);
                                        match kind {
                                            DiseqGuardKind::Distinct => {
                                                z4_sat::Literal::negative(sat_var)
                                            }
                                            DiseqGuardKind::Eq => {
                                                z4_sat::Literal::positive(sat_var)
                                            }
                                        }
                                    })
                                });
                            // If bounds prove the variable is pinned to the excluded value,
                            // force the disequality guard to be false instead of enumerating
                            // excluded values until we hit the per-variable split cap.
                            if !can_lt && !can_gt {
                                if let Some(g) = guard_lit {
                                    sat.add_clause(vec![g]);
                                    continue;
                                }
                            }
                            if !can_lt && can_gt {
                                if let Some(g) = guard_lit {
                                    sat.add_clause(vec![g, z4_sat::Literal::positive(gt_sat_var)]);
                                } else {
                                    sat.add_clause(vec![z4_sat::Literal::positive(gt_sat_var)]);
                                }
                            } else if !can_gt && can_lt {
                                if let Some(g) = guard_lit {
                                    sat.add_clause(vec![g, z4_sat::Literal::positive(lt_sat_var)]);
                                } else {
                                    sat.add_clause(vec![z4_sat::Literal::positive(lt_sat_var)]);
                                }
                            } else {
                                // Add splitting clause: (lt_atom OR gt_atom)
                                if let Some(g) = guard_lit {
                                    sat.add_clause(vec![
                                        g,
                                        z4_sat::Literal::positive(lt_sat_var),
                                        z4_sat::Literal::positive(gt_sat_var),
                                    ]);
                                } else {
                                    sat.add_clause(vec![
                                        z4_sat::Literal::positive(lt_sat_var),
                                        z4_sat::Literal::positive(gt_sat_var),
                                    ]);
                                }
                            }

                            if debug {
                                eprintln!(
                                    "[CHC-SMT] Added disequality split clause: lt_var={}, gt_var={}, left_atom={:?}, right_atom={:?}, num_vars={}, var_to_term.len()={}",
                                    lt_var, gt_var, left_atom, right_atom, num_vars, var_to_term.len()
                                );
                                // Check if split atoms are actually in var_to_term
                                let lt_in_map = var_to_term.get(&lt_var);
                                let gt_in_map = var_to_term.get(&gt_var);
                                eprintln!(
                                    "[CHC-SMT]   var_to_term[{}] = {:?}, var_to_term[{}] = {:?}",
                                    lt_var, lt_in_map, gt_var, gt_in_map
                                );
                            }

                            // Continue the DPLL(T) loop
                            continue;
                        }
                        TheoryResult::NeedExpressionSplit(split) => {
                            // Multi-variable disequality split: E != F
                            // Create atoms (E < F) OR (E > F) - avoids infinite value enumeration

                            split_count += 1;
                            if split_count > MAX_SPLITS {
                                if debug {
                                    eprintln!("[CHC-SMT] Exceeded max splits ({})", MAX_SPLITS);
                                }
                                return SmtResult::Unknown;
                            }

                            if debug {
                                eprintln!(
                                    "[CHC-SMT] iter {}: NeedExpressionSplit on disequality {:?}",
                                    dt_iterations, split.disequality_term
                                );
                            }

                            // Extract LHS and RHS from the disequality term
                            // The term is either `(= E F)` (asserted false) or `(distinct E F)` (asserted true)
                            let (lhs, rhs) = match self.terms.get(split.disequality_term) {
                                TermData::App(Symbol::Named(name), args) if args.len() == 2 => {
                                    if name == "=" || name == "distinct" {
                                        (args[0], args[1])
                                    } else {
                                        if debug {
                                            eprintln!("[CHC-SMT] ExpressionSplit: unexpected operator {:?}", name);
                                        }
                                        return SmtResult::Unknown;
                                    }
                                }
                                _ => {
                                    if debug {
                                        eprintln!("[CHC-SMT] ExpressionSplit: cannot parse disequality term {:?}", split.disequality_term);
                                    }
                                    return SmtResult::Unknown;
                                }
                            };

                            // Create (E < F) and (E > F) atoms
                            let lt_atom = self.terms.mk_lt(lhs, rhs);
                            let gt_atom = self.terms.mk_gt(lhs, rhs);

                            if debug {
                                eprintln!(
                                    "[CHC-SMT] Created expression split atoms: lt={:?} ({:?}), gt={:?} ({:?})",
                                    lt_atom, self.terms.get(lt_atom), gt_atom, self.terms.get(gt_atom)
                                );
                            }

                            // Allocate CNF variables for the split atoms
                            let lt_var = *term_to_var.entry(lt_atom).or_insert_with(|| {
                                num_vars += 1;
                                var_to_term.insert(num_vars, lt_atom);
                                num_vars
                            });
                            let gt_var = *term_to_var.entry(gt_atom).or_insert_with(|| {
                                num_vars += 1;
                                var_to_term.insert(num_vars, gt_atom);
                                num_vars
                            });

                            // Tell the SAT solver about new variables
                            sat.ensure_num_vars(num_vars as usize);

                            let lt_sat_var = z4_sat::Variable(lt_var - 1);
                            let gt_sat_var = z4_sat::Variable(gt_var - 1);

                            // Add splitting clause: (E < F) OR (E > F)
                            sat.add_clause(vec![
                                z4_sat::Literal::positive(lt_sat_var),
                                z4_sat::Literal::positive(gt_sat_var),
                            ]);

                            if debug {
                                eprintln!(
                                    "[CHC-SMT] Added expression split clause: lt_var={}, gt_var={}",
                                    lt_var, gt_var
                                );
                            }

                            // Continue the DPLL(T) loop
                            continue;
                        }
                    }
                }
                z4_sat::AssumeResult::Unsat(core) => {
                    if let Some(map) = &assumption_map {
                        let mut conjuncts = Vec::new();
                        for lit in core {
                            if let Some(expr) = map.get(&lit) {
                                conjuncts.push(expr.clone());
                            }
                        }
                        if !conjuncts.is_empty() {
                            return SmtResult::UnsatWithCore(UnsatCore { conjuncts });
                        }
                    }
                    return SmtResult::Unsat;
                }
                z4_sat::AssumeResult::Unknown => return SmtResult::Unknown,
            }
        }
    }

    /// Check satisfiability under assumptions and return an UNSAT core over the assumptions.
    ///
    /// - `background` conjuncts are asserted permanently (unit clauses).
    /// - `assumptions` are passed as SAT assumptions; if UNSAT, a subset is returned as core.
    ///
    /// This is a lightweight building block for inductive UNSAT cores (IUCs) in PDR.
    /// Callers are expected to pass already-normalized formulas (typically no ITE/mod).
    pub fn check_sat_with_assumption_conjuncts(
        &mut self,
        background: &[ChcExpr],
        assumptions: &[ChcExpr],
    ) -> SmtResult {
        use z4_core::{TheoryResult, TheorySolver, Tseitin};
        use z4_lia::LiaSolver;

        if assumptions.is_empty() {
            let bg = background
                .iter()
                .cloned()
                .reduce(ChcExpr::and)
                .unwrap_or(ChcExpr::Bool(true));
            return self.check_sat(&bg);
        }

        let debug = std::env::var("Z4_DEBUG_CHC_SMT").is_ok();

        let bg_terms: Vec<TermId> = background.iter().map(|b| self.convert_expr(b)).collect();
        let assumption_terms: Vec<(ChcExpr, TermId)> = assumptions
            .iter()
            .map(|a| (a.clone(), self.convert_expr(a)))
            .collect();

        let mut tseitin = Tseitin::new(&self.terms);
        for term in bg_terms {
            tseitin.encode_and_assert(term);
        }

        let mut sat_assumptions: Vec<z4_sat::Literal> = Vec::with_capacity(assumption_terms.len());
        let mut assumption_map: FxHashMap<z4_sat::Literal, ChcExpr> = FxHashMap::default();
        for (a, term) in assumption_terms {
            let cnf_lit = tseitin.encode(term, true);
            let sat_lit = cnf_lit_to_sat_lit(cnf_lit);
            sat_assumptions.push(sat_lit);
            assumption_map.insert(sat_lit, a);
        }

        let mut sat = z4_sat::Solver::new(tseitin.num_vars() as usize);
        for clause in tseitin.all_clauses() {
            let lits: Vec<z4_sat::Literal> = clause
                .0
                .iter()
                .map(|&lit| cnf_lit_to_sat_lit(lit))
                .collect();
            sat.add_clause(lits);
        }

        let mut term_to_var = tseitin.term_to_var().clone();
        let mut var_to_term = tseitin.var_to_term().clone();
        let mut num_vars = tseitin.num_vars();

        let mut split_count: usize = 0;
        const MAX_SPLITS: usize = 10000;
        let mut dt_iterations: usize = 0;

        loop {
            dt_iterations += 1;
            if dt_iterations > 10_000 {
                if debug {
                    eprintln!("[CHC-SMT] Exceeded max iterations (assumption mode)");
                }
                return SmtResult::Unknown;
            }

            match sat.solve_with_assumptions(&sat_assumptions) {
                z4_sat::AssumeResult::Sat(model) => {
                    let mut lia = LiaSolver::new(&self.terms);
                    for (&var_idx, &term_id) in &var_to_term {
                        if !is_theory_atom(&self.terms, term_id) {
                            continue;
                        }
                        let sat_var = z4_sat::Variable(var_idx - 1);
                        if let Some(value) = model.get(sat_var.0 as usize) {
                            lia.assert_literal(term_id, *value);
                        }
                    }

                    match lia.check() {
                        TheoryResult::Sat => return SmtResult::Sat(FxHashMap::default()),
                        TheoryResult::Unknown => return SmtResult::Unknown,
                        TheoryResult::Unsat(conflict) => {
                            if conflict.is_empty() {
                                return SmtResult::Unsat;
                            }
                            let clause: Vec<z4_sat::Literal> = conflict
                                .iter()
                                .filter_map(|lit| {
                                    term_to_var.get(&lit.term).map(|&cnf_var| {
                                        let sat_var = z4_sat::Variable(cnf_var - 1);
                                        if lit.value {
                                            z4_sat::Literal::negative(sat_var)
                                        } else {
                                            z4_sat::Literal::positive(sat_var)
                                        }
                                    })
                                })
                                .collect();
                            if clause.is_empty() {
                                return SmtResult::Unsat;
                            }
                            sat.add_clause(clause);
                        }
                        TheoryResult::NeedSplit(split) => {
                            split_count += 1;
                            if split_count > MAX_SPLITS {
                                return SmtResult::Unknown;
                            }

                            let floor_term = self.terms.mk_int(split.floor.clone());
                            let ceil_term = self.terms.mk_int(split.ceil.clone());
                            let le_atom = self.terms.mk_le(split.variable, floor_term);
                            let ge_atom = self.terms.mk_ge(split.variable, ceil_term);

                            let le_var = *term_to_var.entry(le_atom).or_insert_with(|| {
                                num_vars += 1;
                                var_to_term.insert(num_vars, le_atom);
                                num_vars
                            });
                            let ge_var = *term_to_var.entry(ge_atom).or_insert_with(|| {
                                num_vars += 1;
                                var_to_term.insert(num_vars, ge_atom);
                                num_vars
                            });

                            sat.ensure_num_vars(num_vars as usize);

                            let le_sat_var = z4_sat::Variable(le_var - 1);
                            let ge_sat_var = z4_sat::Variable(ge_var - 1);
                            sat.add_clause(vec![
                                z4_sat::Literal::positive(le_sat_var),
                                z4_sat::Literal::positive(ge_sat_var),
                            ]);
                            continue;
                        }
                        TheoryResult::NeedDisequlitySplit(split) => {
                            split_count += 1;
                            if split_count > MAX_SPLITS {
                                return SmtResult::Unknown;
                            }

                            let (left_atom, right_atom) = if self.terms.sort(split.variable)
                                == &Sort::Int
                                && split.excluded_value.is_integer()
                            {
                                let excluded = split.excluded_value.numer().clone();
                                let le_bound = self.terms.mk_int(&excluded - BigInt::from(1));
                                let ge_bound = self.terms.mk_int(&excluded + BigInt::from(1));
                                (
                                    self.terms.mk_le(split.variable, le_bound),
                                    self.terms.mk_ge(split.variable, ge_bound),
                                )
                            } else {
                                let excluded_term =
                                    self.terms.mk_rational(split.excluded_value.clone());
                                (
                                    self.terms.mk_lt(split.variable, excluded_term),
                                    self.terms.mk_gt(split.variable, excluded_term),
                                )
                            };

                            let lt_var = *term_to_var.entry(left_atom).or_insert_with(|| {
                                num_vars += 1;
                                var_to_term.insert(num_vars, left_atom);
                                num_vars
                            });
                            let gt_var = *term_to_var.entry(right_atom).or_insert_with(|| {
                                num_vars += 1;
                                var_to_term.insert(num_vars, right_atom);
                                num_vars
                            });

                            sat.ensure_num_vars(num_vars as usize);

                            let lt_sat_var = z4_sat::Variable(lt_var - 1);
                            let gt_sat_var = z4_sat::Variable(gt_var - 1);
                            let guard_lit = self
                                .find_diseq_guard_atom(
                                    split.variable,
                                    &split.excluded_value,
                                    &model,
                                    var_to_term.iter(),
                                )
                                .and_then(|(guard_term, kind)| {
                                    term_to_var.get(&guard_term).copied().map(|cnf_var| {
                                        let sat_var = z4_sat::Variable(cnf_var - 1);
                                        match kind {
                                            DiseqGuardKind::Distinct => {
                                                z4_sat::Literal::negative(sat_var)
                                            }
                                            DiseqGuardKind::Eq => {
                                                z4_sat::Literal::positive(sat_var)
                                            }
                                        }
                                    })
                                });
                            // Same pruning as in `check_sat`: if current bounds make one branch
                            // infeasible, force the other branch as a unit clause.
                            let (can_lt, can_gt) = if split.excluded_value.is_integer() {
                                let excluded_i64 =
                                    bigint_to_i64_saturating(split.excluded_value.numer());
                                let (lb, ub) = self.infer_int_bounds_from_sat_model(
                                    split.variable,
                                    &model,
                                    var_to_term.iter(),
                                );
                                let lt_threshold = excluded_i64.checked_sub(1);
                                let gt_threshold = excluded_i64.checked_add(1);

                                let can_lt =
                                    lt_threshold.is_some_and(|t| lb.is_none_or(|l| l <= t));
                                let can_gt =
                                    gt_threshold.is_some_and(|t| ub.is_none_or(|u| u >= t));
                                (can_lt, can_gt)
                            } else {
                                (true, true)
                            };

                            if !can_lt && !can_gt {
                                if let Some(g) = guard_lit {
                                    sat.add_clause(vec![g]);
                                    continue;
                                }
                            }
                            if !can_lt && can_gt {
                                if let Some(g) = guard_lit {
                                    sat.add_clause(vec![g, z4_sat::Literal::positive(gt_sat_var)]);
                                } else {
                                    sat.add_clause(vec![z4_sat::Literal::positive(gt_sat_var)]);
                                }
                            } else if !can_gt && can_lt {
                                if let Some(g) = guard_lit {
                                    sat.add_clause(vec![g, z4_sat::Literal::positive(lt_sat_var)]);
                                } else {
                                    sat.add_clause(vec![z4_sat::Literal::positive(lt_sat_var)]);
                                }
                            } else if let Some(g) = guard_lit {
                                sat.add_clause(vec![
                                    g,
                                    z4_sat::Literal::positive(lt_sat_var),
                                    z4_sat::Literal::positive(gt_sat_var),
                                ]);
                            } else {
                                sat.add_clause(vec![
                                    z4_sat::Literal::positive(lt_sat_var),
                                    z4_sat::Literal::positive(gt_sat_var),
                                ]);
                            }
                            continue;
                        }
                        TheoryResult::NeedExpressionSplit(split) => {
                            // Multi-variable disequality split: E != F
                            // Create atoms (E < F) OR (E > F)
                            split_count += 1;
                            if split_count > MAX_SPLITS {
                                return SmtResult::Unknown;
                            }

                            // Extract LHS and RHS from the disequality term
                            let (lhs, rhs) = match self.terms.get(split.disequality_term) {
                                TermData::App(Symbol::Named(name), args) if args.len() == 2 => {
                                    if name == "=" || name == "distinct" {
                                        (args[0], args[1])
                                    } else {
                                        return SmtResult::Unknown;
                                    }
                                }
                                _ => return SmtResult::Unknown,
                            };

                            // Create (E < F) and (E > F) atoms
                            let lt_atom = self.terms.mk_lt(lhs, rhs);
                            let gt_atom = self.terms.mk_gt(lhs, rhs);

                            // Allocate CNF variables
                            let lt_var = *term_to_var.entry(lt_atom).or_insert_with(|| {
                                num_vars += 1;
                                var_to_term.insert(num_vars, lt_atom);
                                num_vars
                            });
                            let gt_var = *term_to_var.entry(gt_atom).or_insert_with(|| {
                                num_vars += 1;
                                var_to_term.insert(num_vars, gt_atom);
                                num_vars
                            });

                            sat.ensure_num_vars(num_vars as usize);

                            let lt_sat_var = z4_sat::Variable(lt_var - 1);
                            let gt_sat_var = z4_sat::Variable(gt_var - 1);

                            // Add splitting clause: (E < F) OR (E > F)
                            sat.add_clause(vec![
                                z4_sat::Literal::positive(lt_sat_var),
                                z4_sat::Literal::positive(gt_sat_var),
                            ]);
                            continue;
                        }
                    }
                }
                z4_sat::AssumeResult::Unsat(core) => {
                    let mut conjuncts = Vec::new();
                    for lit in core {
                        if let Some(expr) = assumption_map.get(&lit) {
                            conjuncts.push(expr.clone());
                        }
                    }
                    if !conjuncts.is_empty() {
                        return SmtResult::UnsatWithCore(UnsatCore { conjuncts });
                    }
                    return SmtResult::Unsat;
                }
                z4_sat::AssumeResult::Unknown => return SmtResult::Unknown,
            }
        }
    }

    /// Get the value of a term from the model (recursive evaluation of compound expressions)
    fn get_term_value(
        &self,
        term: TermId,
        values: &FxHashMap<String, SmtValue>,
        lia_model: &Option<z4_lia::LiaModel>,
    ) -> Option<i64> {
        use z4_core::term::{Constant, TermData};

        match self.terms.get(term) {
            TermData::Const(c) => {
                // Integer constant
                match c {
                    Constant::Int(n) => Some(bigint_to_i64_saturating(n)),
                    _ => None,
                }
            }
            TermData::Var(name, _) => {
                // Check if we have a value for this variable
                if let Some(SmtValue::Int(v)) = values.get(name) {
                    return Some(*v);
                }
                // Try LIA model
                if let Some(m) = lia_model {
                    if let Some(v) = m.values.get(&term) {
                        return Some(bigint_to_i64_saturating(v));
                    }
                }
                None
            }
            TermData::App(sym, args) => {
                // Evaluate compound expressions recursively
                let sym_name = sym.name();
                match sym_name {
                    "+" => {
                        // Addition: evaluate all args and sum
                        let mut sum: i64 = 0;
                        for &arg in args {
                            let val = self.get_term_value(arg, values, lia_model)?;
                            sum = sum.saturating_add(val);
                        }
                        Some(sum)
                    }
                    "-" => {
                        // Subtraction: first arg minus rest
                        if args.is_empty() {
                            return None;
                        }
                        let first = self.get_term_value(args[0], values, lia_model)?;
                        if args.len() == 1 {
                            // Unary negation
                            return Some(first.saturating_neg());
                        }
                        let mut result = first;
                        for &arg in args.iter().skip(1) {
                            let val = self.get_term_value(arg, values, lia_model)?;
                            result = result.saturating_sub(val);
                        }
                        Some(result)
                    }
                    "*" => {
                        // Multiplication: product of all args
                        let mut product: i64 = 1;
                        for &arg in args {
                            let val = self.get_term_value(arg, values, lia_model)?;
                            product = product.saturating_mul(val);
                        }
                        Some(product)
                    }
                    "div" | "intdiv" => {
                        // Integer division
                        if args.len() != 2 {
                            return None;
                        }
                        let lhs = self.get_term_value(args[0], values, lia_model)?;
                        let rhs = self.get_term_value(args[1], values, lia_model)?;
                        if rhs == 0 {
                            return None;
                        }
                        Some(lhs / rhs)
                    }
                    "mod" => {
                        // Modulo
                        if args.len() != 2 {
                            return None;
                        }
                        let lhs = self.get_term_value(args[0], values, lia_model)?;
                        let rhs = self.get_term_value(args[1], values, lia_model)?;
                        if rhs == 0 {
                            return None;
                        }
                        Some(lhs % rhs)
                    }
                    "neg" => {
                        // Negation
                        if args.len() != 1 {
                            return None;
                        }
                        let val = self.get_term_value(args[0], values, lia_model)?;
                        Some(val.saturating_neg())
                    }
                    _ => {
                        // Unknown operation - try LIA model directly
                        if let Some(m) = lia_model {
                            if let Some(v) = m.values.get(&term) {
                                return Some(bigint_to_i64_saturating(v));
                            }
                        }
                        None
                    }
                }
            }
            TermData::Ite(_cond, then_term, else_term) => {
                // If-then-else: need to evaluate condition as boolean
                // For integer results, try to evaluate based on condition value
                // First try LIA model
                if let Some(m) = lia_model {
                    if let Some(v) = m.values.get(&term) {
                        return Some(bigint_to_i64_saturating(v));
                    }
                }
                // Try both branches
                if let Some(then_val) = self.get_term_value(*then_term, values, lia_model) {
                    return Some(then_val);
                }
                self.get_term_value(*else_term, values, lia_model)
            }
            TermData::Not(_) => {
                // Boolean negation - not an integer value
                None
            }
            TermData::Let(_, body) => {
                // Let expression - evaluate the body
                self.get_term_value(*body, values, lia_model)
            }
        }
    }

    /// Check if `formula` implies `conclusion` (i.e., formula /\ not(conclusion) is UNSAT)
    pub fn check_implies(&mut self, formula: &ChcExpr, conclusion: &ChcExpr) -> bool {
        // Check: formula /\ not(conclusion) is UNSAT?
        let not_conclusion = ChcExpr::not(conclusion.clone());
        let query = ChcExpr::and(formula.clone(), not_conclusion);

        matches!(
            self.check_sat(&query),
            SmtResult::Unsat | SmtResult::UnsatWithCore(_)
        )
    }

    /// Check if `formula` is satisfiable and return a model if so
    pub fn get_model(&mut self, formula: &ChcExpr) -> Option<FxHashMap<String, SmtValue>> {
        match self.check_sat(formula) {
            SmtResult::Sat(model) => Some(model),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DiseqGuardKind {
    Distinct,
    Eq,
}

/// Convert a DIMACS-style CNF literal to a z4-sat Literal
fn cnf_lit_to_sat_lit(lit: z4_core::CnfLit) -> z4_sat::Literal {
    let var = z4_sat::Variable(lit.unsigned_abs() - 1);
    if lit > 0 {
        z4_sat::Literal::positive(var)
    } else {
        z4_sat::Literal::negative(var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_bool_constant() {
        let mut ctx = SmtContext::new();

        let true_expr = ChcExpr::Bool(true);
        let term = ctx.convert_expr(&true_expr);
        assert!(ctx.terms.is_true(term));

        let false_expr = ChcExpr::Bool(false);
        let term = ctx.convert_expr(&false_expr);
        assert!(ctx.terms.is_false(term));
    }

    #[test]
    fn test_convert_variable() {
        let mut ctx = SmtContext::new();

        let x = ChcVar::new("x", ChcSort::Int);
        let x_expr = ChcExpr::var(x.clone());

        let term1 = ctx.convert_expr(&x_expr);
        let term2 = ctx.convert_expr(&x_expr);

        // Same variable should map to same term
        assert_eq!(term1, term2);
    }

    #[test]
    fn test_convert_arithmetic() {
        let mut ctx = SmtContext::new();

        // x + 1
        let x = ChcVar::new("x", ChcSort::Int);
        let expr = ChcExpr::add(ChcExpr::var(x), ChcExpr::int(1));

        let term = ctx.convert_expr(&expr);
        // Just verify it converts without error
        assert!(term.0 > 0);
    }

    #[test]
    fn test_convert_comparison() {
        let mut ctx = SmtContext::new();

        // x < 5
        let x = ChcVar::new("x", ChcSort::Int);
        let expr = ChcExpr::lt(ChcExpr::var(x), ChcExpr::int(5));

        let term = ctx.convert_expr(&expr);
        // Just verify it converts without error
        assert!(term.0 > 0);
    }

    #[test]
    fn test_check_sat_trivial() {
        let mut ctx = SmtContext::new();

        // true is SAT
        let expr = ChcExpr::Bool(true);
        let result = ctx.check_sat(&expr);
        assert!(matches!(result, SmtResult::Sat(_)));

        // false is UNSAT
        let expr = ChcExpr::Bool(false);
        ctx.reset();
        let result = ctx.check_sat(&expr);
        assert!(matches!(result, SmtResult::Unsat));
    }

    #[test]
    fn test_check_sat_simple_constraint() {
        let mut ctx = SmtContext::new();

        // x = 5 is SAT
        let x = ChcVar::new("x", ChcSort::Int);
        let expr = ChcExpr::eq(ChcExpr::var(x), ChcExpr::int(5));
        let result = ctx.check_sat(&expr);
        assert!(matches!(result, SmtResult::Sat(_)));
    }

    #[test]
    fn test_check_sat_model_with_arith_ite() {
        let mut ctx = SmtContext::new();

        // Base constraints (mirrors a shape that shows up in CHC predecessor queries):
        //   E = 1
        //   A + 1 = 4            => A = 3
        //   ite(E <= A, B + 1, B) = 5  => since E<=A, B = 4
        let a = ChcVar::new("A", ChcSort::Int);
        let b = ChcVar::new("B", ChcSort::Int);
        let e = ChcVar::new("E", ChcSort::Int);

        let expr = ChcExpr::and(
            ChcExpr::eq(
                ChcExpr::add(ChcExpr::int(1), ChcExpr::var(a.clone())),
                ChcExpr::int(4),
            ),
            ChcExpr::and(
                ChcExpr::eq(ChcExpr::var(e.clone()), ChcExpr::int(1)),
                ChcExpr::eq(
                    ChcExpr::ite(
                        ChcExpr::le(ChcExpr::var(e), ChcExpr::var(a)),
                        ChcExpr::add(ChcExpr::int(1), ChcExpr::var(b.clone())),
                        ChcExpr::var(b.clone()),
                    ),
                    ChcExpr::int(5),
                ),
            ),
        );

        let result = ctx.check_sat(&expr);
        let SmtResult::Sat(model) = result else {
            panic!("expected SAT, got {result:?}");
        };

        assert_eq!(model.get("A"), Some(&SmtValue::Int(3)));
        assert_eq!(model.get("E"), Some(&SmtValue::Int(1)));
        assert_eq!(model.get("B"), Some(&SmtValue::Int(4)));
    }

    #[test]
    fn test_check_sat_model_preserves_linear_bindings_under_context() {
        let mut ctx = SmtContext::new();

        let a = ChcVar::new("A", ChcSort::Int);
        let b = ChcVar::new("B", ChcSort::Int);
        let e = ChcVar::new("E", ChcSort::Int);

        let base = ChcExpr::and(
            ChcExpr::not(ChcExpr::le(
                ChcExpr::mul(ChcExpr::int(5), ChcExpr::var(e.clone())),
                ChcExpr::var(a.clone()),
            )),
            ChcExpr::and(
                ChcExpr::eq(
                    ChcExpr::add(ChcExpr::int(1), ChcExpr::var(a.clone())),
                    ChcExpr::int(4),
                ),
                ChcExpr::and(
                    ChcExpr::eq(ChcExpr::var(e.clone()), ChcExpr::int(1)),
                    ChcExpr::eq(
                        ChcExpr::ite(
                            ChcExpr::le(ChcExpr::var(e.clone()), ChcExpr::var(a.clone())),
                            ChcExpr::add(ChcExpr::int(1), ChcExpr::var(b.clone())),
                            ChcExpr::var(b.clone()),
                        ),
                        ChcExpr::int(5),
                    ),
                ),
            ),
        );

        let context = ChcExpr::and(
            ChcExpr::ge(ChcExpr::var(a.clone()), ChcExpr::var(e.clone())),
            ChcExpr::and(
                ChcExpr::le(ChcExpr::var(e.clone()), ChcExpr::var(a.clone())),
                ChcExpr::not(ChcExpr::and(
                    ChcExpr::eq(ChcExpr::var(b.clone()), ChcExpr::int(0)),
                    ChcExpr::ge(ChcExpr::var(a.clone()), ChcExpr::int(4)),
                )),
            ),
        );

        let expr = ChcExpr::and(base, context);
        let result = ctx.check_sat(&expr);
        let SmtResult::Sat(model) = result else {
            panic!("expected SAT, got {result:?}");
        };

        assert_eq!(model.get("A"), Some(&SmtValue::Int(3)));
        assert_eq!(model.get("E"), Some(&SmtValue::Int(1)));
        assert_eq!(model.get("B"), Some(&SmtValue::Int(4)));
    }

    #[test]
    fn test_check_sat_contradiction() {
        let mut ctx = SmtContext::new();

        // x = 5 /\ x = 6 is UNSAT
        let x = ChcVar::new("x", ChcSort::Int);
        let eq5 = ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(5));
        let eq6 = ChcExpr::eq(ChcExpr::var(x), ChcExpr::int(6));
        let expr = ChcExpr::and(eq5, eq6);
        let result = ctx.check_sat(&expr);
        assert!(matches!(result, SmtResult::Unsat));
    }

    #[test]
    fn test_check_implies_bool() {
        let mut ctx = SmtContext::new();

        // a implies (a \/ b)
        let a = ChcVar::new("a", ChcSort::Bool);
        let b = ChcVar::new("b", ChcSort::Bool);
        let formula = ChcExpr::var(a.clone());
        let conclusion = ChcExpr::or(ChcExpr::var(a), ChcExpr::var(b));

        let result = ctx.check_implies(&formula, &conclusion);
        assert!(result);
    }

    #[test]
    fn test_check_implies_bool_false() {
        let mut ctx = SmtContext::new();

        // a does NOT imply b
        let a = ChcVar::new("a", ChcSort::Bool);
        let b = ChcVar::new("b", ChcSort::Bool);
        let formula = ChcExpr::var(a);
        let conclusion = ChcExpr::var(b);

        let result = ctx.check_implies(&formula, &conclusion);
        assert!(!result);
    }

    #[test]
    fn test_lia_distinct_constraint() {
        let mut ctx = SmtContext::new();

        // (x < 1) /\ (x >= 0) /\ (x != 0) should be UNSAT for integers
        let x = ChcVar::new("x", ChcSort::Int);
        let x_var = ChcExpr::var(x.clone());

        // x < 1
        let lt1 = ChcExpr::lt(x_var.clone(), ChcExpr::int(1));
        // x >= 0 (not (x < 0))
        let ge0 = ChcExpr::not(ChcExpr::lt(x_var.clone(), ChcExpr::int(0)));
        // x != 0 (not (x = 0))
        let ne0 = ChcExpr::not(ChcExpr::eq(x_var.clone(), ChcExpr::int(0)));

        let formula = ChcExpr::and(ChcExpr::and(lt1, ge0), ne0);
        eprintln!("Testing formula: {}", formula);

        let result = ctx.check_sat(&formula);
        eprintln!("Result: {:?}", result);

        // For integers: x < 1 /\ x >= 0 /\ x != 0 has NO solution (only x=0 satisfies first two)
        assert!(
            matches!(result, SmtResult::Unsat | SmtResult::UnsatWithCore(_)),
            "Expected UNSAT but got {:?}",
            result
        );
    }

    #[test]
    fn test_lia_distinct_with_slack() {
        let mut ctx = SmtContext::new();

        // (B < 1) /\ (B != 0) should be SAT for integers (e.g., B = -1)
        // This is the query that was returning Unknown in CHC PDR
        let b = ChcVar::new("B", ChcSort::Int);
        let b_var = ChcExpr::var(b.clone());

        // B < 1 (normalized to B <= 0 for integers)
        let lt1 = ChcExpr::lt(b_var.clone(), ChcExpr::int(1));
        // B != 0
        let ne0 = ChcExpr::ne(b_var.clone(), ChcExpr::int(0));

        let formula = ChcExpr::and(lt1, ne0);
        eprintln!("Testing formula: {}", formula);

        let result = ctx.check_sat(&formula);
        eprintln!("Result: {:?}", result);

        // B <= 0 /\ B != 0 means B <= -1, which is SAT (e.g., B = -1)
        assert!(
            matches!(result, SmtResult::Sat(_)),
            "Expected SAT but got {:?}",
            result
        );
    }

    #[test]
    fn test_two_var_disequality() {
        let mut ctx = SmtContext::new();

        // D != E should be SAT (e.g., D = 0, E = 1)
        // This is a 2-variable disequality that was failing in CHC PDR
        let d = ChcVar::new("D", ChcSort::Int);
        let e = ChcVar::new("E", ChcSort::Int);
        let d_var = ChcExpr::var(d);
        let e_var = ChcExpr::var(e);

        let formula = ChcExpr::ne(d_var, e_var);
        eprintln!("Testing formula: {}", formula);

        let result = ctx.check_sat(&formula);
        eprintln!("Result: {:?}", result);

        // D != E is SAT (infinitely many solutions)
        assert!(
            matches!(result, SmtResult::Sat(_)),
            "Expected SAT but got {:?}",
            result
        );
    }

    #[test]
    fn test_compound_expr_equality() {
        let mut ctx = SmtContext::new();

        // Test: (x + 1) = (y + 1) /\ x = 5 /\ y = 6 should be UNSAT (6 != 7)
        let x = ChcVar::new("x", ChcSort::Int);
        let y = ChcVar::new("y", ChcSort::Int);

        let x_plus_1 = ChcExpr::add(ChcExpr::var(x.clone()), ChcExpr::int(1));
        let y_plus_1 = ChcExpr::add(ChcExpr::var(y.clone()), ChcExpr::int(1));
        let eq_compound = ChcExpr::eq(x_plus_1, y_plus_1);

        let x_eq_5 = ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(5));
        let y_eq_6 = ChcExpr::eq(ChcExpr::var(y.clone()), ChcExpr::int(6));

        let formula = ChcExpr::and(ChcExpr::and(eq_compound, x_eq_5), y_eq_6);
        let result = ctx.check_sat(&formula);

        // (5+1) != (6+1), so should be UNSAT
        assert!(
            matches!(result, SmtResult::Unsat),
            "Expected UNSAT but got {:?}",
            result
        );
    }

    #[test]
    fn test_compound_expr_equality_sat() {
        let mut ctx = SmtContext::new();

        // Test: (x + 1) = (y + 1) /\ x = 5 /\ y = 5 should be SAT (6 == 6)
        let x = ChcVar::new("x", ChcSort::Int);
        let y = ChcVar::new("y", ChcSort::Int);

        let x_plus_1 = ChcExpr::add(ChcExpr::var(x.clone()), ChcExpr::int(1));
        let y_plus_1 = ChcExpr::add(ChcExpr::var(y.clone()), ChcExpr::int(1));
        let eq_compound = ChcExpr::eq(x_plus_1, y_plus_1);

        let x_eq_5 = ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(5));
        let y_eq_5 = ChcExpr::eq(ChcExpr::var(y.clone()), ChcExpr::int(5));

        let formula = ChcExpr::and(ChcExpr::and(eq_compound, x_eq_5), y_eq_5);
        let result = ctx.check_sat(&formula);

        // (5+1) == (5+1), so should be SAT
        assert!(
            matches!(result, SmtResult::Sat(_)),
            "Expected SAT but got {:?}",
            result
        );
    }

    #[test]
    fn test_compound_expr_subtraction() {
        let mut ctx = SmtContext::new();

        // Test: (x - 3) = 2 /\ x = 5 should be SAT
        let x = ChcVar::new("x", ChcSort::Int);

        let x_minus_3 = ChcExpr::sub(ChcExpr::var(x.clone()), ChcExpr::int(3));
        let eq_expr = ChcExpr::eq(x_minus_3, ChcExpr::int(2));
        let x_eq_5 = ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(5));

        let formula = ChcExpr::and(eq_expr, x_eq_5);
        let result = ctx.check_sat(&formula);

        // 5 - 3 = 2, so should be SAT
        assert!(
            matches!(result, SmtResult::Sat(_)),
            "Expected SAT but got {:?}",
            result
        );
    }

    #[test]
    fn test_compound_expr_multiplication() {
        let mut ctx = SmtContext::new();

        // Test: (x * 2) = 10 /\ x = 5 should be SAT
        let x = ChcVar::new("x", ChcSort::Int);

        let x_times_2 = ChcExpr::mul(ChcExpr::var(x.clone()), ChcExpr::int(2));
        let eq_expr = ChcExpr::eq(x_times_2, ChcExpr::int(10));
        let x_eq_5 = ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(5));

        let formula = ChcExpr::and(eq_expr, x_eq_5);
        let result = ctx.check_sat(&formula);

        // 5 * 2 = 10, so should be SAT
        assert!(
            matches!(result, SmtResult::Sat(_)),
            "Expected SAT but got {:?}",
            result
        );
    }

    #[test]
    fn test_nested_compound_expr() {
        let mut ctx = SmtContext::new();

        // Test: ((x + 1) * 2) = 12 /\ x = 5 should be SAT
        let x = ChcVar::new("x", ChcSort::Int);

        let x_plus_1 = ChcExpr::add(ChcExpr::var(x.clone()), ChcExpr::int(1));
        let x_plus_1_times_2 = ChcExpr::mul(x_plus_1, ChcExpr::int(2));
        let eq_expr = ChcExpr::eq(x_plus_1_times_2, ChcExpr::int(12));
        let x_eq_5 = ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(5));

        let formula = ChcExpr::and(eq_expr, x_eq_5);
        let result = ctx.check_sat(&formula);

        // (5 + 1) * 2 = 12, so should be SAT
        assert!(
            matches!(result, SmtResult::Sat(_)),
            "Expected SAT but got {:?}",
            result
        );
    }

    // =========================================================================
    // Mod elimination tests
    // =========================================================================

    #[test]
    fn test_mod_elimination_basic() {
        // Test that mod elimination works for a simple case:
        // x = 7 /\ (mod x 3) = 1 should be SAT (7 mod 3 = 1)
        let mut ctx = SmtContext::new();

        let x = ChcVar::new("x", ChcSort::Int);
        let x_eq_7 = ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(7));
        let x_mod_3 = ChcExpr::Op(
            ChcOp::Mod,
            vec![
                std::sync::Arc::new(ChcExpr::var(x)),
                std::sync::Arc::new(ChcExpr::int(3)),
            ],
        );
        let mod_eq_1 = ChcExpr::eq(x_mod_3, ChcExpr::int(1));
        let formula = ChcExpr::and(x_eq_7, mod_eq_1);

        let result = ctx.check_sat(&formula);
        assert!(
            matches!(result, SmtResult::Sat(_)),
            "Expected SAT for x=7, x mod 3 = 1, got {:?}",
            result
        );
    }

    #[test]
    fn test_mod_elimination_unsat() {
        // Test that mod elimination correctly detects UNSAT:
        // x = 7 /\ (mod x 3) = 2 should be UNSAT (7 mod 3 = 1, not 2)
        let mut ctx = SmtContext::new();

        let x = ChcVar::new("x", ChcSort::Int);
        let x_eq_7 = ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(7));
        let x_mod_3 = ChcExpr::Op(
            ChcOp::Mod,
            vec![
                std::sync::Arc::new(ChcExpr::var(x)),
                std::sync::Arc::new(ChcExpr::int(3)),
            ],
        );
        let mod_eq_2 = ChcExpr::eq(x_mod_3, ChcExpr::int(2));
        let formula = ChcExpr::and(x_eq_7, mod_eq_2);

        let result = ctx.check_sat(&formula);
        assert!(
            matches!(result, SmtResult::Unsat),
            "Expected UNSAT for x=7, x mod 3 = 2, got {:?}",
            result
        );
    }

    #[test]
    fn test_mod_elimination_constraint_sat() {
        // Test mod constraint satisfaction:
        // (mod x 2) = 0 should be SAT (x can be any even number)
        let mut ctx = SmtContext::new();

        let x = ChcVar::new("x", ChcSort::Int);
        let x_mod_2 = ChcExpr::Op(
            ChcOp::Mod,
            vec![
                std::sync::Arc::new(ChcExpr::var(x.clone())),
                std::sync::Arc::new(ChcExpr::int(2)),
            ],
        );
        let mod_eq_0 = ChcExpr::eq(x_mod_2, ChcExpr::int(0));
        // Add bound to make it finite: 0 <= x <= 10
        let x_ge_0 = ChcExpr::ge(ChcExpr::var(x.clone()), ChcExpr::int(0));
        let x_le_10 = ChcExpr::le(ChcExpr::var(x), ChcExpr::int(10));
        let formula = ChcExpr::and(ChcExpr::and(mod_eq_0, x_ge_0), x_le_10);

        let result = ctx.check_sat(&formula);
        assert!(
            matches!(result, SmtResult::Sat(_)),
            "Expected SAT for x mod 2 = 0 with 0 <= x <= 10, got {:?}",
            result
        );
    }

    #[test]
    fn test_mod_elimination_no_solution() {
        // Test impossible constraint:
        // (mod x 5) = 10 should be UNSAT (remainder must be < divisor)
        let mut ctx = SmtContext::new();

        let x = ChcVar::new("x", ChcSort::Int);
        let x_mod_5 = ChcExpr::Op(
            ChcOp::Mod,
            vec![
                std::sync::Arc::new(ChcExpr::var(x.clone())),
                std::sync::Arc::new(ChcExpr::int(5)),
            ],
        );
        let mod_eq_10 = ChcExpr::eq(x_mod_5, ChcExpr::int(10));
        // Add bound to ensure solver doesn't run forever: 0 <= x <= 100
        let x_ge_0 = ChcExpr::ge(ChcExpr::var(x.clone()), ChcExpr::int(0));
        let x_le_100 = ChcExpr::le(ChcExpr::var(x), ChcExpr::int(100));
        let formula = ChcExpr::and(ChcExpr::and(mod_eq_10, x_ge_0), x_le_100);

        let result = ctx.check_sat(&formula);
        assert!(
            matches!(result, SmtResult::Unsat | SmtResult::UnsatWithCore(_)),
            "Expected UNSAT for x mod 5 = 10 (remainder cannot be >= divisor), got {:?}",
            result
        );
    }

    #[test]
    fn test_mod_eliminate_expr() {
        // Unit test for the eliminate_mod method itself
        let x = ChcVar::new("x", ChcSort::Int);
        let x_mod_3 = ChcExpr::Op(
            ChcOp::Mod,
            vec![
                std::sync::Arc::new(ChcExpr::var(x.clone())),
                std::sync::Arc::new(ChcExpr::int(3)),
            ],
        );
        let original = ChcExpr::eq(x_mod_3, ChcExpr::int(1));
        let eliminated = original.eliminate_mod();

        // The eliminated expression should be an AND of multiple constraints
        // (the original comparison + definitional constraints)
        match &eliminated {
            ChcExpr::Op(ChcOp::And, args) => {
                // Should have multiple conjuncts:
                // 1. x = k*q + r
                // 2. r >= 0
                // 3. r < |k|
                // 4. r = 1 (the original constraint rewritten)
                assert!(
                    args.len() >= 4,
                    "Expected at least 4 conjuncts after mod elimination, got {}",
                    args.len()
                );
            }
            _ => panic!(
                "Expected AND expression after mod elimination, got {:?}",
                eliminated
            ),
        }
    }

    // =========================================================================
    // ITE / div-by-zero / mod-by-zero elimination tests
    // =========================================================================

    #[test]
    fn test_ite_elimination_sat() {
        // x = (ite b 1 0) /\ b = true /\ x = 1 should be SAT.
        // Without ite elimination, `(= x (ite ...))` is not a linear arithmetic atom.
        let mut ctx = SmtContext::new();

        let b = ChcVar::new("b", ChcSort::Bool);
        let x = ChcVar::new("x", ChcSort::Int);

        let b_true = ChcExpr::eq(ChcExpr::var(b.clone()), ChcExpr::bool_const(true));
        let x_eq_ite = ChcExpr::eq(
            ChcExpr::var(x.clone()),
            ChcExpr::ite(ChcExpr::var(b.clone()), ChcExpr::int(1), ChcExpr::int(0)),
        );
        let x_eq_1 = ChcExpr::eq(ChcExpr::var(x), ChcExpr::int(1));

        let formula = ChcExpr::and(ChcExpr::and(b_true, x_eq_ite), x_eq_1);
        let result = ctx.check_sat(&formula);
        assert!(
            matches!(result, SmtResult::Sat(_)),
            "Expected SAT but got {:?}",
            result
        );
    }

    #[test]
    fn test_ite_elimination_unsat() {
        // x = (ite b 1 0) /\ b = true /\ x = 0 should be UNSAT.
        let mut ctx = SmtContext::new();

        let b = ChcVar::new("b", ChcSort::Bool);
        let x = ChcVar::new("x", ChcSort::Int);

        let b_true = ChcExpr::eq(ChcExpr::var(b.clone()), ChcExpr::bool_const(true));
        let x_eq_ite = ChcExpr::eq(
            ChcExpr::var(x.clone()),
            ChcExpr::ite(ChcExpr::var(b.clone()), ChcExpr::int(1), ChcExpr::int(0)),
        );
        let x_eq_0 = ChcExpr::eq(ChcExpr::var(x), ChcExpr::int(0));

        let formula = ChcExpr::and(ChcExpr::and(b_true, x_eq_ite), x_eq_0);
        let result = ctx.check_sat(&formula);
        assert!(
            matches!(result, SmtResult::Unsat | SmtResult::UnsatWithCore(_)),
            "Expected UNSAT but got {:?}",
            result
        );
    }

    #[test]
    fn test_mod_by_zero_semantics() {
        // SMT-LIB: (mod x 0) = x
        let mut ctx = SmtContext::new();

        let x = ChcVar::new("x", ChcSort::Int);
        let x_eq_5 = ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(5));
        let x_mod_0 = ChcExpr::Op(
            ChcOp::Mod,
            vec![
                std::sync::Arc::new(ChcExpr::var(x.clone())),
                std::sync::Arc::new(ChcExpr::int(0)),
            ],
        );

        let mod_eq_5 = ChcExpr::eq(x_mod_0.clone(), ChcExpr::int(5));
        let sat_formula = ChcExpr::and(x_eq_5.clone(), mod_eq_5);
        assert!(matches!(ctx.check_sat(&sat_formula), SmtResult::Sat(_)));

        let mod_eq_6 = ChcExpr::eq(x_mod_0, ChcExpr::int(6));
        let unsat_formula = ChcExpr::and(x_eq_5, mod_eq_6);
        assert!(matches!(
            ctx.check_sat(&unsat_formula),
            SmtResult::Unsat | SmtResult::UnsatWithCore(_)
        ));
    }

    #[test]
    fn test_div_by_zero_semantics() {
        // SMT-LIB: (div x 0) = 0
        let mut ctx = SmtContext::new();

        let x = ChcVar::new("x", ChcSort::Int);
        let x_eq_7 = ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(7));
        let x_div_0 = ChcExpr::Op(
            ChcOp::Div,
            vec![
                std::sync::Arc::new(ChcExpr::var(x)),
                std::sync::Arc::new(ChcExpr::int(0)),
            ],
        );

        let div_eq_0 = ChcExpr::eq(x_div_0.clone(), ChcExpr::int(0));
        let sat_formula = ChcExpr::and(x_eq_7.clone(), div_eq_0);
        assert!(matches!(ctx.check_sat(&sat_formula), SmtResult::Sat(_)));

        let div_eq_1 = ChcExpr::eq(x_div_0, ChcExpr::int(1));
        let unsat_formula = ChcExpr::and(x_eq_7, div_eq_1);
        assert!(matches!(
            ctx.check_sat(&unsat_formula),
            SmtResult::Unsat | SmtResult::UnsatWithCore(_)
        ));
    }
}
