//! K-Induction solver for CHC problems
//!
//! K-Induction is a bounded model checking algorithm that combines:
//! - Base case: Check if error states are reachable in up to k steps
//! - Induction step: Check if k-step induction proves safety
//!
//! This is a port of Golem's Kind engine (200 lines C++).
//!
//! ## Algorithm
//!
//! For each k from 0 to max_k:
//!
//! 1. **Base case**: Init(x₀) ∧ Tr(x₀,x₁) ∧ ... ∧ Tr(xₖ₋₁,xₖ) ∧ Query(xₖ)
//!    - If SAT: found counterexample of length k → UNSAFE
//!
//! 2. **Forward induction**: ¬Query(x₀) ∧ Tr(x₀,x₁) ∧ ¬Query(x₁) ∧ ... ∧ ¬Query(xₖ) ⟹ ¬Query(xₖ₊₁)
//!    - If valid (negation is UNSAT): found k-inductive invariant → SAFE
//!
//! 3. **Backward induction**: Tr(x₀,x₁) ∧ ¬Init(x₁) ∧ ... ∧ Tr(xₖ,xₖ₊₁) ∧ ¬Init(xₖ₊₁) ⟹ ¬Init(x₀)
//!    - If valid (negation is UNSAT): found k-inductive invariant → SAFE
//!
//! ## Applicability
//!
//! K-Induction works best on linear CHC problems (single predicate transition systems).
//! For non-linear problems, we fall back to PDR.

use crate::smt::{SmtContext, SmtResult};
use crate::{ChcExpr, ChcProblem, ChcVar, Model, PredicateId, PredicateInterpretation};

/// K-Induction solver result
#[derive(Debug)]
pub enum KindResult {
    /// Safe: found k-inductive invariant
    Safe { k: usize, invariant: ChcExpr },
    /// Unsafe: found counterexample of length k
    Unsafe { k: usize },
    /// Unknown: reached max_k without conclusion
    Unknown,
    /// Not applicable: problem is not a linear transition system
    NotApplicable,
}

/// K-Induction solver configuration
#[derive(Debug, Clone)]
pub struct KindConfig {
    /// Maximum k to try
    pub max_k: usize,
    /// Enable verbose logging
    pub verbose: bool,
    /// Timeout per SMT query
    pub query_timeout: std::time::Duration,
    /// Total time budget for the entire K-Induction run
    /// If exceeded, returns Unknown and lets PDR take over
    pub total_timeout: std::time::Duration,
}

impl Default for KindConfig {
    fn default() -> Self {
        Self {
            max_k: 50,
            verbose: false,
            query_timeout: std::time::Duration::from_secs(5),
            total_timeout: std::time::Duration::from_secs(30),
        }
    }
}

/// Transition system extracted from a linear CHC problem
struct TransitionSystem {
    /// The single predicate (invariant to synthesize)
    _predicate: PredicateId,
    /// Canonical variables for the predicate
    vars: Vec<ChcVar>,
    /// Initial state constraint: constraint on vars such that Init(vars) holds
    init: ChcExpr,
    /// Transition relation: constraint on (vars, vars') such that Tr(vars, vars') holds
    /// vars are the "current state", vars' are the "next state" (with _next suffix)
    transition: ChcExpr,
    /// Query/bad state constraint: constraint on vars such that Query(vars) → false
    query: ChcExpr,
}

impl TransitionSystem {
    /// Version variable names for timestep k
    /// x becomes x_k
    fn version_var(var: &ChcVar, k: usize) -> ChcVar {
        ChcVar::new(format!("{}_{}", var.name, k), var.sort.clone())
    }

    /// Version an expression for timestep k
    fn version_expr(expr: &ChcExpr, vars: &[ChcVar], k: usize) -> ChcExpr {
        let substitutions: Vec<_> = vars
            .iter()
            .map(|v| (v.clone(), ChcExpr::var(Self::version_var(v, k))))
            .collect();
        expr.substitute(&substitutions)
    }

    /// Create transition constraint from step k to step k+1
    fn transition_at(&self, k: usize) -> ChcExpr {
        // transition is over (vars, vars_next)
        // We need to substitute vars -> vars_k and vars_next -> vars_{k+1}
        let mut substitutions: Vec<_> = self
            .vars
            .iter()
            .map(|v| (v.clone(), ChcExpr::var(Self::version_var(v, k))))
            .collect();

        // Also substitute _next variables
        for v in &self.vars {
            let next_var = ChcVar::new(format!("{}_next", v.name), v.sort.clone());
            substitutions.push((next_var, ChcExpr::var(Self::version_var(v, k + 1))));
        }

        self.transition.substitute(&substitutions)
    }

    /// Create init constraint at step k
    fn init_at(&self, k: usize) -> ChcExpr {
        Self::version_expr(&self.init, &self.vars, k)
    }

    /// Create query constraint at step k
    fn query_at(&self, k: usize) -> ChcExpr {
        Self::version_expr(&self.query, &self.vars, k)
    }

    /// Create ¬query at step k
    fn neg_query_at(&self, k: usize) -> ChcExpr {
        ChcExpr::not(self.query_at(k))
    }

    /// Create ¬init at step k
    fn neg_init_at(&self, k: usize) -> ChcExpr {
        ChcExpr::not(self.init_at(k))
    }
}

/// K-Induction solver
pub struct KindSolver<'a> {
    problem: &'a ChcProblem,
    config: KindConfig,
}

impl<'a> KindSolver<'a> {
    /// Create a new K-Induction solver
    pub fn new(problem: &'a ChcProblem, config: KindConfig) -> Self {
        Self { problem, config }
    }

    /// Check if the problem is a linear transition system (single predicate)
    fn is_transition_system(&self) -> bool {
        // Must have exactly one predicate
        if self.problem.predicates().len() != 1 {
            return false;
        }

        // Must have at least one fact (init), one transition, and one query
        let has_fact = self.problem.facts().count() > 0;
        let has_transition = self.problem.transitions().count() > 0;
        let has_query = self.problem.queries().count() > 0;

        // Each transition must involve only the single predicate
        let pred_id = self.problem.predicates()[0].id;
        let transitions_ok = self.problem.transitions().all(|t| {
            // Body should have exactly one predicate application
            t.body.predicates.len() == 1
                && t.body.predicates[0].0 == pred_id
                // Head should also be the same predicate
                && t.head.predicate_id() == Some(pred_id)
        });

        has_fact && has_transition && has_query && transitions_ok
    }

    /// Extract transition system from the CHC problem
    fn extract_transition_system(&self) -> Option<TransitionSystem> {
        if !self.is_transition_system() {
            return None;
        }

        let pred = &self.problem.predicates()[0];
        let pred_id = pred.id;

        // Create canonical variables
        let vars: Vec<ChcVar> = pred
            .arg_sorts
            .iter()
            .enumerate()
            .map(|(i, sort)| ChcVar::new(format!("v{}", i), sort.clone()))
            .collect();

        // Extract init constraint from fact clauses
        // Fact: constraint => Pred(args)
        // Init = exists args. constraint[vars/args]
        let init = self.extract_init_constraint(pred_id, &vars)?;

        // Extract transition constraint from transition clauses
        // Transition: Pred(args) ∧ constraint => Pred(args')
        // Tr(vars, vars_next) = exists args, args'. vars=args ∧ vars_next=args' ∧ constraint
        let transition = self.extract_transition_constraint(pred_id, &vars)?;

        // Extract query constraint from query clauses
        // Query: Pred(args) ∧ constraint => false
        // Query(vars) = exists args. vars=args ∧ constraint
        let query = self.extract_query_constraint(pred_id, &vars)?;

        Some(TransitionSystem {
            _predicate: pred_id,
            vars,
            init,
            transition,
            query,
        })
    }

    /// Extract init constraint: maps fact clauses to constraint on canonical vars
    fn extract_init_constraint(&self, pred_id: PredicateId, vars: &[ChcVar]) -> Option<ChcExpr> {
        let mut init_conjuncts = Vec::new();

        for fact in self.problem.facts() {
            if fact.head.predicate_id() != Some(pred_id) {
                continue;
            }

            // Get head arguments
            let head_args = match &fact.head {
                crate::ClauseHead::Predicate(_, args) => args,
                _ => continue,
            };

            // Build substitution: head_args[i] should equal vars[i]
            // If head_arg is a variable, substitute it with canonical var
            // If head_arg is an expression, add equality constraint
            let mut constraint = fact.body.constraint.clone().unwrap_or(ChcExpr::Bool(true));

            for (i, head_arg) in head_args.iter().enumerate() {
                if let ChcExpr::Var(v) = head_arg {
                    // Substitute the variable
                    constraint = constraint.substitute(&[(v.clone(), ChcExpr::var(vars[i].clone()))]);
                } else {
                    // Add equality: vars[i] = head_arg
                    let eq = ChcExpr::eq(ChcExpr::var(vars[i].clone()), head_arg.clone());
                    constraint = ChcExpr::and(constraint, eq);
                }
            }

            init_conjuncts.push(constraint);
        }

        if init_conjuncts.is_empty() {
            None
        } else if init_conjuncts.len() == 1 {
            Some(init_conjuncts.pop().unwrap())
        } else {
            // Multiple fact clauses: disjunction
            Some(init_conjuncts.into_iter().reduce(ChcExpr::or).unwrap())
        }
    }

    /// Extract transition constraint: maps transition clauses to constraint on (vars, vars_next)
    fn extract_transition_constraint(&self, pred_id: PredicateId, vars: &[ChcVar]) -> Option<ChcExpr> {
        let mut trans_conjuncts = Vec::new();

        // Create "next" variables
        let next_vars: Vec<ChcVar> = vars
            .iter()
            .map(|v| ChcVar::new(format!("{}_next", v.name), v.sort.clone()))
            .collect();

        for trans in self.problem.transitions() {
            // Body should have the predicate
            let (body_pred, body_args) = trans.body.predicates.first()?;
            if *body_pred != pred_id {
                continue;
            }

            // Head should also be the predicate
            let head_args = match &trans.head {
                crate::ClauseHead::Predicate(p, args) if *p == pred_id => args,
                _ => continue,
            };

            let mut constraint = trans.body.constraint.clone().unwrap_or(ChcExpr::Bool(true));

            // Substitute body args -> vars (current state)
            for (i, body_arg) in body_args.iter().enumerate() {
                if let ChcExpr::Var(v) = body_arg {
                    constraint = constraint.substitute(&[(v.clone(), ChcExpr::var(vars[i].clone()))]);
                }
            }

            // Handle head args -> next_vars (next state)
            for (i, head_arg) in head_args.iter().enumerate() {
                match head_arg {
                    ChcExpr::Var(v) => {
                        // Check if this is a body variable that we already substituted
                        let body_var_idx = body_args.iter().position(|ba| {
                            if let ChcExpr::Var(bv) = ba {
                                bv.name == v.name
                            } else {
                                false
                            }
                        });

                        if let Some(idx) = body_var_idx {
                            // head_arg is a body variable - add equality next_vars[i] = vars[idx]
                            let eq = ChcExpr::eq(
                                ChcExpr::var(next_vars[i].clone()),
                                ChcExpr::var(vars[idx].clone()),
                            );
                            constraint = ChcExpr::and(constraint, eq);
                        } else {
                            // head_arg is a fresh variable - substitute it
                            constraint = constraint
                                .substitute(&[(v.clone(), ChcExpr::var(next_vars[i].clone()))]);
                        }
                    }
                    _ => {
                        // head_arg is an expression - add equality next_vars[i] = expr
                        // But first substitute body vars in the expression
                        let mut expr = head_arg.clone();
                        for (j, body_arg) in body_args.iter().enumerate() {
                            if let ChcExpr::Var(bv) = body_arg {
                                expr = expr.substitute(&[(bv.clone(), ChcExpr::var(vars[j].clone()))]);
                            }
                        }
                        let eq = ChcExpr::eq(ChcExpr::var(next_vars[i].clone()), expr);
                        constraint = ChcExpr::and(constraint, eq);
                    }
                }
            }

            trans_conjuncts.push(constraint);
        }

        if trans_conjuncts.is_empty() {
            None
        } else if trans_conjuncts.len() == 1 {
            Some(trans_conjuncts.pop().unwrap())
        } else {
            // Multiple transition clauses: disjunction
            Some(trans_conjuncts.into_iter().reduce(ChcExpr::or).unwrap())
        }
    }

    /// Extract query constraint: maps query clauses to bad state constraint
    fn extract_query_constraint(&self, pred_id: PredicateId, vars: &[ChcVar]) -> Option<ChcExpr> {
        let mut query_conjuncts = Vec::new();

        for query in self.problem.queries() {
            // Body should have the predicate
            let (body_pred, body_args) = query.body.predicates.first()?;
            if *body_pred != pred_id {
                continue;
            }

            let mut constraint = query.body.constraint.clone().unwrap_or(ChcExpr::Bool(true));

            // Substitute body args -> vars
            for (i, body_arg) in body_args.iter().enumerate() {
                if let ChcExpr::Var(v) = body_arg {
                    constraint = constraint.substitute(&[(v.clone(), ChcExpr::var(vars[i].clone()))]);
                }
            }

            query_conjuncts.push(constraint);
        }

        if query_conjuncts.is_empty() {
            None
        } else if query_conjuncts.len() == 1 {
            Some(query_conjuncts.pop().unwrap())
        } else {
            // Multiple query clauses: disjunction
            Some(query_conjuncts.into_iter().reduce(ChcExpr::or).unwrap())
        }
    }

    /// Run K-Induction algorithm
    pub fn solve(&self) -> KindResult {
        let start_time = std::time::Instant::now();

        // Check if problem is applicable
        let ts = match self.extract_transition_system() {
            Some(ts) => ts,
            None => return KindResult::NotApplicable,
        };

        if self.config.verbose {
            eprintln!("KIND: Extracted transition system with {} vars", ts.vars.len());
            eprintln!("KIND: Init: {}", ts.init);
            eprintln!("KIND: Query: {}", ts.query);
        }

        // Check trivial case: init is empty
        let mut ctx = SmtContext::new();
        match ctx.check_sat_with_timeout(&ts.init, self.config.query_timeout) {
            SmtResult::Unsat | SmtResult::UnsatWithCore(_) => {
                if self.config.verbose {
                    eprintln!("KIND: Init is empty, trivially safe");
                }
                return KindResult::Safe {
                    k: 0,
                    invariant: ChcExpr::Bool(true),
                };
            }
            _ => {}
        }

        // Main k-induction loop
        // We maintain three incremental formulas:
        // - base_formula: Init(x0) ∧ Tr^k for BMC
        // - forward_formula: Query(x0) ∧ Tr ∧ ¬Query(x1) ∧ ... for forward induction
        // - backward_formula: Init(x0) ∧ Tr ∧ ¬Init(x1) ∧ ... for backward induction

        for k in 0..=self.config.max_k {
            // Check total time budget
            if start_time.elapsed() > self.config.total_timeout {
                if self.config.verbose {
                    eprintln!("KIND: Total timeout exceeded after k = {}", k);
                }
                return KindResult::Unknown;
            }

            if self.config.verbose {
                eprintln!("KIND: Trying k = {}", k);
            }

            // Build base case formula: Init(x0) ∧ Tr^k ∧ Query(xk)
            let base_result = self.check_base_case(&ts, k);
            match base_result {
                SmtResult::Sat(_) => {
                    if self.config.verbose {
                        eprintln!("KIND: Found counterexample at k = {}", k);
                    }
                    return KindResult::Unsafe { k };
                }
                SmtResult::Unknown => {
                    if self.config.verbose {
                        eprintln!("KIND: Base case unknown at k = {}", k);
                    }
                    // Continue to try larger k
                }
                SmtResult::Unsat | SmtResult::UnsatWithCore(_) => {
                    if self.config.verbose {
                        eprintln!("KIND: No path of length {} to bad state", k);
                    }
                }
            }

            // Check forward induction: ¬Query(x0) ∧ Tr ∧ ¬Query(x1) ∧ ... ∧ Tr ∧ ¬Query(xk) ⟹ ¬Query(x{k+1})
            // Equivalently: ¬Query(x0) ∧ Tr ∧ ... ∧ ¬Query(xk) ∧ Query(x{k+1}) is UNSAT
            let forward_result = self.check_forward_induction(&ts, k);
            match forward_result {
                SmtResult::Unsat | SmtResult::UnsatWithCore(_) => {
                    if self.config.verbose {
                        eprintln!("KIND: Forward induction succeeded at k = {}", k);
                    }
                    return KindResult::Safe {
                        k,
                        invariant: ChcExpr::not(ts.query.clone()),
                    };
                }
                _ => {}
            }

            // Check backward induction: ¬Init(x{k+1}) ⟸ Tr ∧ ¬Init(x1) ∧ ... ∧ Tr ∧ ¬Init(xk) ∧ Tr ∧ ¬Init(x0)
            // Equivalently: Init(x0) ∧ Tr ∧ ¬Init(x1) ∧ ... ∧ ¬Init(x{k+1}) is UNSAT
            let backward_result = self.check_backward_induction(&ts, k);
            match backward_result {
                SmtResult::Unsat | SmtResult::UnsatWithCore(_) => {
                    if self.config.verbose {
                        eprintln!("KIND: Backward induction succeeded at k = {}", k);
                    }
                    return KindResult::Safe {
                        k,
                        invariant: ts.init.clone(),
                    };
                }
                _ => {}
            }
        }

        KindResult::Unknown
    }

    /// Check base case: Init(x0) ∧ Tr(x0,x1) ∧ ... ∧ Tr(x{k-1},xk) ∧ Query(xk)
    fn check_base_case(&self, ts: &TransitionSystem, k: usize) -> SmtResult {
        let mut ctx = SmtContext::new();

        // Build formula
        let mut conjuncts = vec![ts.init_at(0)];

        for i in 0..k {
            conjuncts.push(ts.transition_at(i));
        }

        conjuncts.push(ts.query_at(k));

        let formula = conjuncts.into_iter().reduce(ChcExpr::and).unwrap();
        ctx.check_sat_with_timeout(&formula, self.config.query_timeout)
    }

    /// Check forward induction: ¬Query(x0) ∧ Tr ∧ ¬Query(x1) ∧ ... ∧ Query(x{k+1}) is UNSAT
    fn check_forward_induction(&self, ts: &TransitionSystem, k: usize) -> SmtResult {
        let mut ctx = SmtContext::new();

        // Build formula: ¬Query(x0) ∧ Tr(x0,x1) ∧ ¬Query(x1) ∧ ... ∧ Tr(xk,x{k+1}) ∧ Query(x{k+1})
        let mut conjuncts = Vec::new();

        for i in 0..=k {
            conjuncts.push(ts.neg_query_at(i));
            if i < k + 1 {
                conjuncts.push(ts.transition_at(i));
            }
        }

        // Add the positive query at the end
        conjuncts.push(ts.query_at(k + 1));

        let formula = conjuncts.into_iter().reduce(ChcExpr::and).unwrap();
        ctx.check_sat_with_timeout(&formula, self.config.query_timeout)
    }

    /// Check backward induction: Init(x0) ∧ Tr ∧ ¬Init(x1) ∧ ... ∧ ¬Init(x{k+1}) is UNSAT
    fn check_backward_induction(&self, ts: &TransitionSystem, k: usize) -> SmtResult {
        let mut ctx = SmtContext::new();

        // Build formula: Init(x0) ∧ Tr(x0,x1) ∧ ¬Init(x1) ∧ ... ∧ Tr(xk,x{k+1}) ∧ ¬Init(x{k+1})
        let mut conjuncts = vec![ts.init_at(0)];

        for i in 0..=k {
            conjuncts.push(ts.transition_at(i));
            conjuncts.push(ts.neg_init_at(i + 1));
        }

        let formula = conjuncts.into_iter().reduce(ChcExpr::and).unwrap();
        ctx.check_sat_with_timeout(&formula, self.config.query_timeout)
    }

    /// Convert K-Induction result to PDR Model format
    pub fn to_model(&self, result: &KindResult) -> Option<Model> {
        match result {
            KindResult::Safe { invariant, .. } => {
                let pred = &self.problem.predicates()[0];
                let vars: Vec<ChcVar> = pred
                    .arg_sorts
                    .iter()
                    .enumerate()
                    .map(|(i, sort)| ChcVar::new(format!("v{}", i), sort.clone()))
                    .collect();

                let mut model = Model::new();
                model.set(pred.id, PredicateInterpretation::new(vars, invariant.clone()));
                Some(model)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChcSort, ClauseBody, ClauseHead, HornClause};

    #[test]
    fn test_simple_counter() {
        // Counter from 0 to 5:
        // x = 0 => Inv(x)
        // Inv(x) ∧ x < 5 => Inv(x+1)
        // Inv(x) ∧ x > 5 => false
        let mut problem = ChcProblem::new();
        let inv = problem.declare_predicate("Inv", vec![ChcSort::Int]);
        let x = ChcVar::new("x", ChcSort::Int);

        // x = 0 => Inv(x)
        problem.add_clause(HornClause::new(
            ClauseBody::constraint(ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(0))),
            ClauseHead::Predicate(inv, vec![ChcExpr::var(x.clone())]),
        ));

        // Inv(x) ∧ x < 5 => Inv(x+1)
        problem.add_clause(HornClause::new(
            ClauseBody::new(
                vec![(inv, vec![ChcExpr::var(x.clone())])],
                Some(ChcExpr::lt(ChcExpr::var(x.clone()), ChcExpr::int(5))),
            ),
            ClauseHead::Predicate(
                inv,
                vec![ChcExpr::add(ChcExpr::var(x.clone()), ChcExpr::int(1))],
            ),
        ));

        // Inv(x) ∧ x > 5 => false
        problem.add_clause(HornClause::new(
            ClauseBody::new(
                vec![(inv, vec![ChcExpr::var(x.clone())])],
                Some(ChcExpr::gt(ChcExpr::var(x.clone()), ChcExpr::int(5))),
            ),
            ClauseHead::False,
        ));

        let config = KindConfig {
            max_k: 10,
            verbose: false,
            query_timeout: std::time::Duration::from_secs(1),
            total_timeout: std::time::Duration::from_secs(10),
        };
        let solver = KindSolver::new(&problem, config);
        let result = solver.solve();

        match result {
            KindResult::Safe { k, .. } => {
                println!("Safe with k = {}", k);
            }
            KindResult::NotApplicable => {
                println!("Not applicable (multi-predicate)");
            }
            other => {
                println!("Result: {:?}", other);
            }
        }
    }

    #[test]
    fn test_unsafe_counter() {
        // Counter that can reach x > 5:
        // x = 0 => Inv(x)
        // Inv(x) => Inv(x+1)  (no guard!)
        // Inv(x) ∧ x > 5 => false
        let mut problem = ChcProblem::new();
        let inv = problem.declare_predicate("Inv", vec![ChcSort::Int]);
        let x = ChcVar::new("x", ChcSort::Int);

        // x = 0 => Inv(x)
        problem.add_clause(HornClause::new(
            ClauseBody::constraint(ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(0))),
            ClauseHead::Predicate(inv, vec![ChcExpr::var(x.clone())]),
        ));

        // Inv(x) => Inv(x+1)
        problem.add_clause(HornClause::new(
            ClauseBody::new(vec![(inv, vec![ChcExpr::var(x.clone())])], None),
            ClauseHead::Predicate(
                inv,
                vec![ChcExpr::add(ChcExpr::var(x.clone()), ChcExpr::int(1))],
            ),
        ));

        // Inv(x) ∧ x > 5 => false
        problem.add_clause(HornClause::new(
            ClauseBody::new(
                vec![(inv, vec![ChcExpr::var(x.clone())])],
                Some(ChcExpr::gt(ChcExpr::var(x.clone()), ChcExpr::int(5))),
            ),
            ClauseHead::False,
        ));

        let config = KindConfig {
            max_k: 10,
            verbose: false,
            query_timeout: std::time::Duration::from_secs(1),
            total_timeout: std::time::Duration::from_secs(10),
        };
        let solver = KindSolver::new(&problem, config);
        let result = solver.solve();

        match result {
            KindResult::Unsafe { k } => {
                println!("Unsafe at k = {}", k);
                assert!(k <= 6, "Should find counterexample at k <= 6");
            }
            other => {
                panic!("Expected Unsafe, got {:?}", other);
            }
        }
    }
}
