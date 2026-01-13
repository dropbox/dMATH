//! Z3 solver integration with timeout support and model extraction
//!
//! This module provides a high-level interface to Z3 for checking
//! satisfiability of TLA+ expressions.

use num_bigint::BigInt;
use std::collections::HashMap;
use std::time::Duration;
use tla_core::ast::Expr;
use tla_core::span::Spanned;
use z3::{Config, Context, Model, SatResult, Solver};

use crate::error::{SmtError, SmtResult};
use crate::translate::{SmtTranslator, Sort};

/// Value extracted from an SMT model
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtValue {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(BigInt),
    /// Unknown/uninterpreted value
    Unknown(String),
}

impl std::fmt::Display for SmtValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SmtValue::Bool(b) => write!(f, "{}", b),
            SmtValue::Int(i) => write!(f, "{}", i),
            SmtValue::Unknown(s) => write!(f, "{}", s),
        }
    }
}

/// A model extracted from a satisfiable SMT query
#[derive(Debug, Clone)]
pub struct SmtModel {
    /// Variable assignments
    pub assignments: HashMap<String, SmtValue>,
}

impl SmtModel {
    /// Get the value of a variable
    pub fn get(&self, name: &str) -> Option<&SmtValue> {
        self.assignments.get(name)
    }

    /// Check if the model is empty
    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    /// Get the number of assignments
    pub fn len(&self) -> usize {
        self.assignments.len()
    }
}

impl std::fmt::Display for SmtModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Model:")?;
        for (name, value) in &self.assignments {
            writeln!(f, "  {} = {}", name, value)?;
        }
        Ok(())
    }
}

/// Result of an SMT check
#[derive(Debug)]
pub enum SmtCheckResult {
    /// Formula is satisfiable, with a model
    Sat(SmtModel),
    /// Formula is unsatisfiable
    Unsat,
    /// Result is unknown (possibly due to timeout or incompleteness)
    Unknown(String),
}

impl SmtCheckResult {
    /// Check if the result is satisfiable
    pub fn is_sat(&self) -> bool {
        matches!(self, SmtCheckResult::Sat(_))
    }

    /// Check if the result is unsatisfiable
    pub fn is_unsat(&self) -> bool {
        matches!(self, SmtCheckResult::Unsat)
    }

    /// Check if the result is unknown
    pub fn is_unknown(&self) -> bool {
        matches!(self, SmtCheckResult::Unknown(_))
    }

    /// Get the model if satisfiable
    pub fn model(&self) -> Option<&SmtModel> {
        match self {
            SmtCheckResult::Sat(m) => Some(m),
            _ => None,
        }
    }
}

/// SMT solver context for checking TLA+ expressions
pub struct SmtContext {
    cfg: Config,
    timeout_ms: Option<u64>,
}

impl Default for SmtContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SmtContext {
    /// Create a new SMT context with default settings
    pub fn new() -> Self {
        Self {
            cfg: Config::new(),
            timeout_ms: None,
        }
    }

    /// Set the timeout for SMT queries
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_ms = Some(timeout.as_millis() as u64);
        self
    }

    /// Set the timeout in milliseconds
    pub fn set_timeout_ms(&mut self, ms: u64) {
        self.timeout_ms = Some(ms);
    }

    /// Check if an expression is satisfiable
    ///
    /// Returns `Sat(model)` if satisfiable, `Unsat` if unsatisfiable,
    /// or `Unknown` if the solver cannot determine satisfiability.
    pub fn check_sat(
        &self,
        expr: &Spanned<Expr>,
        vars: &[(String, Sort)],
    ) -> SmtResult<SmtCheckResult> {
        let ctx = Context::new(&self.cfg);
        let mut translator = SmtTranslator::new(&ctx);

        // Declare variables
        for (name, sort) in vars {
            translator.declare_var(name, sort.clone())?;
        }

        // Translate expression
        let z3_expr = translator.translate_bool(expr)?;

        // Create solver and add constraint
        let solver = Solver::new(&ctx);

        // Set timeout if configured
        if let Some(ms) = self.timeout_ms {
            let mut params = z3::Params::new(&ctx);
            params.set_u32("timeout", ms as u32);
            solver.set_params(&params);
        }

        solver.assert(&z3_expr);

        // Check satisfiability
        match solver.check() {
            SatResult::Sat => {
                let model = solver
                    .get_model()
                    .ok_or_else(|| SmtError::ModelError("no model available".to_string()))?;
                let smt_model = self.extract_model(&model, &translator)?;
                Ok(SmtCheckResult::Sat(smt_model))
            }
            SatResult::Unsat => Ok(SmtCheckResult::Unsat),
            SatResult::Unknown => {
                let reason = solver
                    .get_reason_unknown()
                    .unwrap_or_else(|| "unknown".to_string());
                Ok(SmtCheckResult::Unknown(reason))
            }
        }
    }

    /// Check if an expression is valid (always true)
    ///
    /// An expression is valid if its negation is unsatisfiable.
    pub fn check_valid(
        &self,
        expr: &Spanned<Expr>,
        vars: &[(String, Sort)],
    ) -> SmtResult<SmtCheckResult> {
        // To check validity, we check if ~expr is unsat
        let negated = Spanned::new(Expr::Not(Box::new(expr.clone())), expr.span);

        match self.check_sat(&negated, vars)? {
            SmtCheckResult::Sat(model) => {
                // Negation is sat, so original is not valid - return counterexample
                Ok(SmtCheckResult::Sat(model))
            }
            SmtCheckResult::Unsat => {
                // Negation is unsat, so original is valid
                Ok(SmtCheckResult::Unsat) // Using Unsat to mean "valid" here
            }
            SmtCheckResult::Unknown(reason) => Ok(SmtCheckResult::Unknown(reason)),
        }
    }

    /// Extract a model from Z3's model
    fn extract_model(&self, model: &Model, translator: &SmtTranslator) -> SmtResult<SmtModel> {
        let mut assignments = HashMap::new();

        for (name, (sort, z3_const)) in translator.vars() {
            let value = match sort {
                Sort::Bool => {
                    if let Some(b) = z3_const.as_bool() {
                        if let Some(val) = model.eval(&b, true) {
                            if let Some(b_val) = val.as_bool() {
                                SmtValue::Bool(b_val)
                            } else {
                                SmtValue::Unknown(val.to_string())
                            }
                        } else {
                            continue; // Variable not in model
                        }
                    } else {
                        continue;
                    }
                }
                Sort::Int => {
                    if let Some(i) = z3_const.as_int() {
                        if let Some(val) = model.eval(&i, true) {
                            if let Some(i_val) = val.as_i64() {
                                SmtValue::Int(BigInt::from(i_val))
                            } else {
                                // Try to parse as BigInt from string
                                SmtValue::Unknown(val.to_string())
                            }
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                Sort::Uninterpreted(_) => {
                    if let Some(val) = model.eval(z3_const, true) {
                        SmtValue::Unknown(val.to_string())
                    } else {
                        continue;
                    }
                }
            };
            assignments.insert(name.clone(), value);
        }

        Ok(SmtModel { assignments })
    }

    /// Check if a constraint system is satisfiable, with multiple assertions
    pub fn check_constraints(
        &self,
        constraints: &[Spanned<Expr>],
        vars: &[(String, Sort)],
    ) -> SmtResult<SmtCheckResult> {
        let ctx = Context::new(&self.cfg);
        let mut translator = SmtTranslator::new(&ctx);

        // Declare variables
        for (name, sort) in vars {
            translator.declare_var(name, sort.clone())?;
        }

        // Create solver
        let solver = Solver::new(&ctx);

        // Set timeout if configured
        if let Some(ms) = self.timeout_ms {
            let mut params = z3::Params::new(&ctx);
            params.set_u32("timeout", ms as u32);
            solver.set_params(&params);
        }

        // Add all constraints
        for constraint in constraints {
            let z3_expr = translator.translate_bool(constraint)?;
            solver.assert(&z3_expr);
        }

        // Check satisfiability
        match solver.check() {
            SatResult::Sat => {
                let model = solver
                    .get_model()
                    .ok_or_else(|| SmtError::ModelError("no model available".to_string()))?;
                let smt_model = self.extract_model(&model, &translator)?;
                Ok(SmtCheckResult::Sat(smt_model))
            }
            SatResult::Unsat => Ok(SmtCheckResult::Unsat),
            SatResult::Unknown => {
                let reason = solver
                    .get_reason_unknown()
                    .unwrap_or_else(|| "unknown".to_string());
                Ok(SmtCheckResult::Unknown(reason))
            }
        }
    }

    /// Prove an implication: given assumptions, does the conclusion hold?
    pub fn prove_implication(
        &self,
        assumptions: &[Spanned<Expr>],
        conclusion: &Spanned<Expr>,
        vars: &[(String, Sort)],
    ) -> SmtResult<SmtCheckResult> {
        let ctx = Context::new(&self.cfg);
        let mut translator = SmtTranslator::new(&ctx);

        // Declare variables
        for (name, sort) in vars {
            translator.declare_var(name, sort.clone())?;
        }

        // Create solver
        let solver = Solver::new(&ctx);

        // Set timeout if configured
        if let Some(ms) = self.timeout_ms {
            let mut params = z3::Params::new(&ctx);
            params.set_u32("timeout", ms as u32);
            solver.set_params(&params);
        }

        // Add assumptions
        for assumption in assumptions {
            let z3_assumption = translator.translate_bool(assumption)?;
            solver.assert(&z3_assumption);
        }

        // Add negation of conclusion
        let z3_conclusion = translator.translate_bool(conclusion)?;
        solver.assert(&z3_conclusion.not());

        // Check: if unsat, then the implication is valid
        match solver.check() {
            SatResult::Sat => {
                // Found counterexample: assumptions hold but conclusion doesn't
                let model = solver
                    .get_model()
                    .ok_or_else(|| SmtError::ModelError("no model available".to_string()))?;
                let smt_model = self.extract_model(&model, &translator)?;
                Ok(SmtCheckResult::Sat(smt_model))
            }
            SatResult::Unsat => {
                // No counterexample: implication is valid
                Ok(SmtCheckResult::Unsat)
            }
            SatResult::Unknown => {
                let reason = solver
                    .get_reason_unknown()
                    .unwrap_or_else(|| "unknown".to_string());
                Ok(SmtCheckResult::Unknown(reason))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::span::{FileId, Span};

    fn span() -> Span {
        Span::new(FileId(0), 0, 0)
    }

    fn spanned<T>(node: T) -> Spanned<T> {
        Spanned::new(node, span())
    }

    #[test]
    fn test_check_sat_simple() {
        let ctx = SmtContext::new();

        // x > 5 /\ x < 10 is satisfiable
        let expr = spanned(Expr::And(
            Box::new(spanned(Expr::Gt(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(5)))),
            ))),
            Box::new(spanned(Expr::Lt(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(10)))),
            ))),
        ));

        let vars = vec![("x".to_string(), Sort::Int)];
        let result = ctx.check_sat(&expr, &vars).unwrap();

        assert!(result.is_sat());
        let model = result.model().unwrap();
        if let Some(SmtValue::Int(x)) = model.get("x") {
            assert!(*x > BigInt::from(5));
            assert!(*x < BigInt::from(10));
        } else {
            panic!("Expected Int value for x");
        }
    }

    #[test]
    fn test_check_sat_unsat() {
        let ctx = SmtContext::new();

        // x > 5 /\ x < 3 is unsatisfiable
        let expr = spanned(Expr::And(
            Box::new(spanned(Expr::Gt(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(5)))),
            ))),
            Box::new(spanned(Expr::Lt(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(3)))),
            ))),
        ));

        let vars = vec![("x".to_string(), Sort::Int)];
        let result = ctx.check_sat(&expr, &vars).unwrap();

        assert!(result.is_unsat());
    }

    #[test]
    fn test_check_constraints() {
        let ctx = SmtContext::new();

        // x = y + 1, y = 5 => x = 6
        let constraints = vec![
            spanned(Expr::Eq(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Add(
                    Box::new(spanned(Expr::Ident("y".to_string()))),
                    Box::new(spanned(Expr::Int(BigInt::from(1)))),
                ))),
            )),
            spanned(Expr::Eq(
                Box::new(spanned(Expr::Ident("y".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(5)))),
            )),
        ];

        let vars = vec![("x".to_string(), Sort::Int), ("y".to_string(), Sort::Int)];

        let result = ctx.check_constraints(&constraints, &vars).unwrap();
        assert!(result.is_sat());

        let model = result.model().unwrap();
        assert_eq!(model.get("x"), Some(&SmtValue::Int(BigInt::from(6))));
        assert_eq!(model.get("y"), Some(&SmtValue::Int(BigInt::from(5))));
    }

    #[test]
    fn test_prove_implication() {
        let ctx = SmtContext::new();

        // Prove: x > 5 => x > 3
        let assumption = spanned(Expr::Gt(
            Box::new(spanned(Expr::Ident("x".to_string()))),
            Box::new(spanned(Expr::Int(BigInt::from(5)))),
        ));
        let conclusion = spanned(Expr::Gt(
            Box::new(spanned(Expr::Ident("x".to_string()))),
            Box::new(spanned(Expr::Int(BigInt::from(3)))),
        ));

        let vars = vec![("x".to_string(), Sort::Int)];
        let result = ctx
            .prove_implication(&[assumption], &conclusion, &vars)
            .unwrap();

        // Should be unsat (meaning the implication is valid)
        assert!(result.is_unsat());
    }

    #[test]
    fn test_prove_implication_invalid() {
        let ctx = SmtContext::new();

        // Try to prove: x > 3 => x > 5 (this is INVALID)
        let assumption = spanned(Expr::Gt(
            Box::new(spanned(Expr::Ident("x".to_string()))),
            Box::new(spanned(Expr::Int(BigInt::from(3)))),
        ));
        let conclusion = spanned(Expr::Gt(
            Box::new(spanned(Expr::Ident("x".to_string()))),
            Box::new(spanned(Expr::Int(BigInt::from(5)))),
        ));

        let vars = vec![("x".to_string(), Sort::Int)];
        let result = ctx
            .prove_implication(&[assumption], &conclusion, &vars)
            .unwrap();

        // Should be sat (counterexample exists, e.g., x = 4)
        assert!(result.is_sat());
        let model = result.model().unwrap();
        if let Some(SmtValue::Int(x)) = model.get("x") {
            assert!(*x > BigInt::from(3));
            assert!(*x <= BigInt::from(5));
        }
    }

    #[test]
    fn test_check_with_timeout() {
        let ctx = SmtContext::new().with_timeout(Duration::from_secs(5));

        // Simple satisfiable formula
        let expr = spanned(Expr::Gt(
            Box::new(spanned(Expr::Ident("x".to_string()))),
            Box::new(spanned(Expr::Int(BigInt::from(0)))),
        ));

        let vars = vec![("x".to_string(), Sort::Int)];
        let result = ctx.check_sat(&expr, &vars).unwrap();

        assert!(result.is_sat());
    }

    #[test]
    fn test_boolean_constraints() {
        let ctx = SmtContext::new();

        // (a \/ b) /\ ~a => b must hold
        let constraint = spanned(Expr::And(
            Box::new(spanned(Expr::Or(
                Box::new(spanned(Expr::Ident("a".to_string()))),
                Box::new(spanned(Expr::Ident("b".to_string()))),
            ))),
            Box::new(spanned(Expr::Not(Box::new(spanned(Expr::Ident(
                "a".to_string(),
            )))))),
        ));

        let vars = vec![("a".to_string(), Sort::Bool), ("b".to_string(), Sort::Bool)];

        let result = ctx.check_sat(&constraint, &vars).unwrap();
        assert!(result.is_sat());

        let model = result.model().unwrap();
        // a must be false, b must be true
        assert_eq!(model.get("a"), Some(&SmtValue::Bool(false)));
        assert_eq!(model.get("b"), Some(&SmtValue::Bool(true)));
    }

    #[test]
    fn test_mixed_types() {
        let ctx = SmtContext::new();

        // flag => x > 0, flag = true, x = 5
        let constraints = vec![
            spanned(Expr::Implies(
                Box::new(spanned(Expr::Ident("flag".to_string()))),
                Box::new(spanned(Expr::Gt(
                    Box::new(spanned(Expr::Ident("x".to_string()))),
                    Box::new(spanned(Expr::Int(BigInt::from(0)))),
                ))),
            )),
            spanned(Expr::Ident("flag".to_string())),
            spanned(Expr::Eq(
                Box::new(spanned(Expr::Ident("x".to_string()))),
                Box::new(spanned(Expr::Int(BigInt::from(5)))),
            )),
        ];

        let vars = vec![
            ("flag".to_string(), Sort::Bool),
            ("x".to_string(), Sort::Int),
        ];

        let result = ctx.check_constraints(&constraints, &vars).unwrap();
        assert!(result.is_sat());

        let model = result.model().unwrap();
        assert_eq!(model.get("flag"), Some(&SmtValue::Bool(true)));
        assert_eq!(model.get("x"), Some(&SmtValue::Int(BigInt::from(5))));
    }

    #[test]
    fn test_model_display() {
        let mut assignments = HashMap::new();
        assignments.insert("x".to_string(), SmtValue::Int(BigInt::from(42)));
        assignments.insert("flag".to_string(), SmtValue::Bool(true));

        let model = SmtModel { assignments };
        let display = format!("{}", model);

        assert!(display.contains("x = 42"));
        assert!(display.contains("flag = true"));
    }
}
