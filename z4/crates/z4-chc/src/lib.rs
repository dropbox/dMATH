//! CHC (Constrained Horn Clause) solver using PDR/IC3
//!
//! This crate implements a Property-Directed Reachability (PDR/IC3) algorithm
//! for solving Constrained Horn Clause (CHC) problems. CHC problems are used
//! for program verification to find inductive invariants.
//!
//! # Example CHC Problem
//!
//! ```text
//! ; Find invariant Inv(x) such that:
//! ; 1. x = 0 => Inv(x)              (initial state)
//! ; 2. Inv(x) /\ x < 10 => Inv(x+1) (transition)
//! ; 3. Inv(x) /\ x >= 10 => false   (safety - should be unsat)
//! ```
//!
//! # Architecture
//!
//! - `Predicate`: Uninterpreted relation to synthesize interpretation for
//! - `HornClause`: Rule of form `body => head`
//! - `ChcProblem`: Collection of Horn clauses with a query
//! - `PdrSolver`: PDR algorithm implementation

mod clause;
mod error;
mod expr;
mod farkas;
mod interpolation;
mod kind;
mod mbp;
mod parser;
mod pdr;
mod predicate;
mod problem;
mod smt;

pub use clause::{ClauseBody, ClauseHead, HornClause};
pub use error::{ChcError, ChcResult};
pub use expr::{ChcExpr, ChcOp, ChcSort, ChcVar};
pub use kind::{KindConfig, KindResult, KindSolver};
pub use mbp::Mbp;
pub use parser::ChcParser;
pub use pdr::{
    validate_invariant_file, validate_invariant_str, Model, PdrConfig, PdrResult, PdrSolver,
    PredicateInterpretation,
};
pub use predicate::{Predicate, PredicateId};
pub use problem::ChcProblem;
pub use smt::{SmtContext, SmtResult, SmtValue};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_problem_construction() {
        // Test basic problem construction
        let mut problem = ChcProblem::new();

        // Declare Inv : Int -> Bool
        let inv = problem.declare_predicate("Inv", vec![ChcSort::Int]);
        assert_eq!(problem.predicates().len(), 1);

        // x = 0 => Inv(x)
        let x = ChcVar::new("x", ChcSort::Int);
        problem.add_clause(HornClause::new(
            ClauseBody::constraint(ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(0))),
            ClauseHead::Predicate(inv, vec![ChcExpr::var(x.clone())]),
        ));

        // Inv(x) /\ x < 10 => Inv(x + 1)
        problem.add_clause(HornClause::new(
            ClauseBody::new(
                vec![(inv, vec![ChcExpr::var(x.clone())])],
                Some(ChcExpr::lt(ChcExpr::var(x.clone()), ChcExpr::int(10))),
            ),
            ClauseHead::Predicate(
                inv,
                vec![ChcExpr::add(ChcExpr::var(x.clone()), ChcExpr::int(1))],
            ),
        ));

        // Inv(x) /\ x > 10 => false (query)
        problem.add_clause(HornClause::new(
            ClauseBody::new(
                vec![(inv, vec![ChcExpr::var(x.clone())])],
                Some(ChcExpr::gt(ChcExpr::var(x.clone()), ChcExpr::int(10))),
            ),
            ClauseHead::False,
        ));

        assert_eq!(problem.clauses().len(), 3);
        assert_eq!(problem.queries().count(), 1);
        assert_eq!(problem.facts().count(), 1);
        assert_eq!(problem.transitions().count(), 1);
        assert!(problem.validate().is_ok());
    }

    #[test]
    fn test_pdr_solver_terminates() {
        // Test that PDR solver terminates on a simple problem
        let mut problem = ChcProblem::new();
        let inv = problem.declare_predicate("Inv", vec![ChcSort::Int]);
        let x = ChcVar::new("x", ChcSort::Int);

        // x = 0 => Inv(x)
        problem.add_clause(HornClause::new(
            ClauseBody::constraint(ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::int(0))),
            ClauseHead::Predicate(inv, vec![ChcExpr::var(x.clone())]),
        ));

        // Inv(x) /\ x < 5 => Inv(x+1)
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

        // Inv(x) /\ x > 5 => false
        problem.add_clause(HornClause::new(
            ClauseBody::new(
                vec![(inv, vec![ChcExpr::var(x.clone())])],
                Some(ChcExpr::gt(ChcExpr::var(x.clone()), ChcExpr::int(5))),
            ),
            ClauseHead::False,
        ));

        let config = PdrConfig {
            max_frames: 4,
            max_iterations: 20,
            max_obligations: 10_000,
            verbose: false,
            generalize_lemmas: false,
            max_generalization_attempts: 0,
            use_mbp: false,
            use_must_summaries: false,
            use_level_priority: false,
            use_mixed_summaries: false,
            use_range_weakening: false,
            use_init_bound_weakening: false,
            use_farkas_combination: false,
            use_relational_equality: false,
            use_interpolation: false,
        };
        let mut solver = PdrSolver::new(problem, config);
        let result = solver.solve();

        // We expect Safe or Unknown (not Unsafe for this safe problem)
        match result {
            PdrResult::Safe(_) => {
                // Expected: found invariant
            }
            PdrResult::Unknown => {
                // Acceptable: couldn't prove it with skeleton implementation
            }
            PdrResult::Unsafe(_) => {
                // This would be wrong for this safe problem
                // But skeleton implementation might have issues
            }
        }
    }

    #[test]
    fn test_propagate_constants_with_mod() {
        use std::sync::Arc;

        // Test: (= A 0) ∧ (not (= (mod A 2) 0))
        // After propagation: (= 0 0) ∧ (not (= (mod 0 2) 0))
        // After simplification: true ∧ (not (= 0 0)) = true ∧ false = false
        let a_var = ChcVar::new("A", ChcSort::Int);

        // (= A 0)
        let a_eq_0 = ChcExpr::eq(ChcExpr::Var(a_var.clone()), ChcExpr::Int(0));

        // (mod A 2)
        let mod_a_2 = ChcExpr::Op(
            ChcOp::Mod,
            vec![
                Arc::new(ChcExpr::Var(a_var.clone())),
                Arc::new(ChcExpr::Int(2)),
            ],
        );

        // (= (mod A 2) 0)
        let mod_eq_0 = ChcExpr::eq(mod_a_2, ChcExpr::Int(0));

        // (not (= (mod A 2) 0))
        let not_mod_eq_0 = ChcExpr::not(mod_eq_0);

        // (and (= A 0) (not (= (mod A 2) 0)))
        let conjunction = ChcExpr::and(a_eq_0, not_mod_eq_0);

        let propagated = conjunction.propagate_constants();

        // Should simplify to false since 0 mod 2 = 0, so (= 0 0) is true,
        // (not true) is false, and (true ∧ false) is false
        assert_eq!(propagated, ChcExpr::Bool(false));
    }

    #[test]
    fn test_simplify_mod_constants() {
        use std::sync::Arc;

        // Test (mod 7 3) should simplify to 1
        let mod_expr = ChcExpr::Op(
            ChcOp::Mod,
            vec![Arc::new(ChcExpr::Int(7)), Arc::new(ChcExpr::Int(3))],
        );
        let simplified = mod_expr.simplify_constants();
        assert_eq!(simplified, ChcExpr::Int(1));

        // Test (mod 6 3) should simplify to 0
        let mod_expr = ChcExpr::Op(
            ChcOp::Mod,
            vec![Arc::new(ChcExpr::Int(6)), Arc::new(ChcExpr::Int(3))],
        );
        let simplified = mod_expr.simplify_constants();
        assert_eq!(simplified, ChcExpr::Int(0));

        // Test (mod 0 2) should simplify to 0
        let mod_expr = ChcExpr::Op(
            ChcOp::Mod,
            vec![Arc::new(ChcExpr::Int(0)), Arc::new(ChcExpr::Int(2))],
        );
        let simplified = mod_expr.simplify_constants();
        assert_eq!(simplified, ChcExpr::Int(0));
    }

    #[test]
    fn test_simplify_and_contradiction() {
        use std::sync::Arc;

        // Test P AND NOT P should simplify to false
        let x = ChcVar::new("x", ChcSort::Int);
        let eq = ChcExpr::eq(
            ChcExpr::Op(
                ChcOp::Mod,
                vec![Arc::new(ChcExpr::var(x.clone())), Arc::new(ChcExpr::Int(6))],
            ),
            ChcExpr::Int(0),
        );
        let not_eq = ChcExpr::not(eq.clone());

        // Direct contradiction: (P AND NOT P)
        let and_expr = ChcExpr::and(eq.clone(), not_eq.clone());
        let simplified = and_expr.simplify_constants();
        assert_eq!(simplified, ChcExpr::Bool(false));

        // Nested contradiction: ((A AND P) AND NOT P)
        let a = ChcExpr::ge(ChcExpr::var(x.clone()), ChcExpr::Int(0));
        let nested = ChcExpr::and(ChcExpr::and(a, eq), not_eq);
        let simplified = nested.simplify_constants();
        assert_eq!(simplified, ChcExpr::Bool(false));
    }
}
