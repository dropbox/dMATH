//! CHC parser for SMT-LIB CHC format
//!
//! This module parses the CHC-COMP and SMT-LIB CHC format, which extends SMT-LIB 2.6
//! with commands for defining Horn clauses:
//!
//! ```text
//! (declare-rel Inv (Int))           ; Declare predicate Inv : Int -> Bool
//! (declare-var x Int)               ; Declare variable x
//! (rule (=> (= x 0) (Inv x)))       ; x = 0 => Inv(x)
//! (rule (=> (and (Inv x) (< x 10)) (Inv (+ x 1))))  ; Inv(x) /\ x < 10 => Inv(x+1)
//! (query Inv)                       ; Check if Inv is satisfiable
//! ```
//!
//! ## Supported Commands
//!
//! - `(set-logic HORN)` - Set logic (ignored but checked)
//! - `(declare-rel <name> (<sorts>))` - Declare a predicate
//! - `(declare-var <name> <sort>)` - Declare a variable
//! - `(declare-fun <name> (<sorts>) <return-sort>)` - Declare a function (predicates return Bool)
//! - `(rule <expr>)` or `(rule (=> <body> <head>))` - Add a Horn clause
//! - `(query <pred>)` - Add a query (safety property)
//! - `(check-sat)` - Solve the CHC problem
//! - `(exit)` - Exit (ignored)

use crate::{
    ChcError, ChcExpr, ChcOp, ChcProblem, ChcResult, ChcSort, ChcVar, ClauseBody, ClauseHead,
    HornClause, PredicateId,
};
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use std::sync::Arc;

/// CHC parser state
pub struct ChcParser {
    /// The CHC problem being built
    problem: ChcProblem,
    /// Declared variables (name -> sort)
    variables: FxHashMap<String, ChcSort>,
    /// Declared predicates (name -> (id, sorts))
    predicates: FxHashMap<String, (PredicateId, Vec<ChcSort>)>,
    /// Current position in input
    pos: usize,
    /// Input string
    input: String,
}

impl ChcParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self {
            problem: ChcProblem::new(),
            variables: FxHashMap::default(),
            predicates: FxHashMap::default(),
            pos: 0,
            input: String::new(),
        }
    }

    /// Parse a CHC file and return the problem
    pub fn parse(input: &str) -> ChcResult<ChcProblem> {
        let mut parser = Self::new();
        parser.input = input.to_string();
        parser.pos = 0;

        while parser.pos < parser.input.len() {
            parser.skip_whitespace_and_comments();
            if parser.pos >= parser.input.len() {
                break;
            }
            parser.parse_command()?;
        }

        Ok(parser.problem)
    }

    /// Parse a single command
    fn parse_command(&mut self) -> ChcResult<()> {
        self.skip_whitespace_and_comments();
        if self.pos >= self.input.len() {
            return Ok(());
        }

        self.expect_char('(')?;
        self.skip_whitespace_and_comments();

        let cmd = self.parse_symbol()?;
        self.skip_whitespace_and_comments();

        match cmd.as_str() {
            "set-logic" => {
                let logic = self.parse_symbol()?;
                // Accept HORN or other logics that might contain CHC
                if !["HORN", "LIA", "LRA", "ALIA", "AUFLIA", "QF_LIA", "QF_LRA"]
                    .contains(&logic.as_str())
                {
                    log::warn!(
                        "Unexpected logic '{}', expecting HORN or arithmetic logic",
                        logic
                    );
                }
            }
            "declare-rel" | "declare-fun" => {
                self.parse_declare_predicate(&cmd)?;
            }
            "declare-var" | "declare-const" => {
                self.parse_declare_var()?;
            }
            "rule" => {
                self.parse_rule()?;
            }
            "query" => {
                self.parse_query()?;
            }
            "check-sat" | "exit" | "set-info" | "set-option" => {
                // Skip until closing paren
                let mut depth = 1;
                while depth > 0 && self.pos < self.input.len() {
                    match self.current_char() {
                        Some('(') => depth += 1,
                        Some(')') => depth -= 1,
                        _ => {}
                    }
                    self.pos += 1;
                }
                return Ok(());
            }
            "assert" => {
                // Parse an assertion (may be a Horn clause)
                self.skip_whitespace_and_comments();
                let expr = self.parse_expr()?;
                self.add_assertion_as_clause(expr)?;
            }
            _ => {
                // Skip unknown command
                log::warn!("Unknown command: {}", cmd);
                let mut depth = 1;
                while depth > 0 && self.pos < self.input.len() {
                    match self.current_char() {
                        Some('(') => depth += 1,
                        Some(')') => depth -= 1,
                        _ => {}
                    }
                    self.pos += 1;
                }
                return Ok(());
            }
        }

        self.skip_whitespace_and_comments();
        self.expect_char(')')?;
        Ok(())
    }

    /// Parse a declare-rel or declare-fun command
    fn parse_declare_predicate(&mut self, cmd: &str) -> ChcResult<()> {
        self.skip_whitespace_and_comments();
        let name = self.parse_symbol()?;
        self.skip_whitespace_and_comments();

        // Parse argument sorts
        self.expect_char('(')?;
        let mut sorts = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            if self.peek_char() == Some(')') {
                break;
            }
            sorts.push(self.parse_sort()?);
        }
        self.expect_char(')')?;

        // For declare-fun, also parse return sort
        if cmd == "declare-fun" {
            self.skip_whitespace_and_comments();
            let ret_sort = self.parse_sort()?;
            if ret_sort != ChcSort::Bool {
                // Not a predicate, just a function - we'll handle it as needed
                log::warn!("Non-predicate function declaration: {}", name);
                return Ok(());
            }
        }

        // Register predicate
        let pred_id = self.problem.declare_predicate(&name, sorts.clone());
        self.predicates.insert(name, (pred_id, sorts));

        Ok(())
    }

    /// Parse a declare-var command
    fn parse_declare_var(&mut self) -> ChcResult<()> {
        self.skip_whitespace_and_comments();
        let name = self.parse_symbol()?;
        self.skip_whitespace_and_comments();
        let sort = self.parse_sort()?;

        self.variables.insert(name, sort);
        Ok(())
    }

    /// Parse a rule command
    fn parse_rule(&mut self) -> ChcResult<()> {
        self.skip_whitespace_and_comments();
        let expr = self.parse_expr()?;

        // Convert expression to Horn clause
        self.add_expr_as_clause(expr)?;
        Ok(())
    }

    /// Parse a query command
    fn parse_query(&mut self) -> ChcResult<()> {
        self.skip_whitespace_and_comments();

        // Query can be a predicate name or an expression
        if self.peek_char() == Some('(') {
            // Expression form
            let expr = self.parse_expr()?;
            // Extract predicates/constraints so the solver can reason about the query predicate.
            // Add as a clause: (preds /\ constraint) => false
            let (preds, constraint) = self.extract_body_parts(&expr);
            let body = if preds.is_empty() && constraint.is_none() {
                ClauseBody::constraint(ChcExpr::Bool(true))
            } else {
                ClauseBody::new(preds, constraint)
            };
            let clause = HornClause::new(body, ClauseHead::False);
            self.problem.add_clause(clause);
        } else {
            // Predicate name form
            let pred_name = self.parse_symbol()?;
            if let Some((pred_id, sorts)) = self.predicates.get(&pred_name).cloned() {
                // Create a query: Pred(x1, ..., xn) => false
                let args: Vec<ChcExpr> = sorts
                    .iter()
                    .enumerate()
                    .map(|(i, sort)| ChcExpr::var(ChcVar::new(format!("_qv{}", i), sort.clone())))
                    .collect();
                let clause = HornClause::new(
                    ClauseBody::new(vec![(pred_id, args)], None),
                    ClauseHead::False,
                );
                self.problem.add_clause(clause);
            } else {
                return Err(ChcError::Parse(format!(
                    "Unknown predicate in query: {}",
                    pred_name
                )));
            }
        }

        Ok(())
    }

    /// Convert an expression to a Horn clause and add it
    fn add_expr_as_clause(&mut self, expr: ChcExpr) -> ChcResult<()> {
        // Pattern: (=> body head) or (forall (...) (=> body head))
        match &expr {
            ChcExpr::Op(ChcOp::Implies, args) if args.len() == 2 => {
                let body = &*args[0];
                let head = &*args[1];
                self.add_implication(body, head)?;
            }
            _ => {
                // Treat as a fact: constraint => true (or constraint itself as head)
                // This handles forall-wrapped expressions or bare predicates
                self.add_fact(expr)?;
            }
        }
        Ok(())
    }

    /// Add an assertion as a clause (for (assert ...) commands)
    fn add_assertion_as_clause(&mut self, expr: ChcExpr) -> ChcResult<()> {
        self.add_expr_as_clause(expr)
    }

    /// Add an implication body => head as a Horn clause
    fn add_implication(&mut self, body: &ChcExpr, head: &ChcExpr) -> ChcResult<()> {
        // Extract predicates and constraints from body
        let (preds, constraint) = self.extract_body_parts(body);

        // Check if head is a predicate application
        let is_head_predicate =
            matches!(head, ChcExpr::PredicateApp(..)) || self.try_extract_predicate(head).is_some();
        let is_head_false = matches!(head, ChcExpr::Bool(false));
        let is_head_true = matches!(head, ChcExpr::Bool(true));

        // If head is true, the clause is trivially satisfied - skip it
        if is_head_true {
            log::debug!("Skipping trivially satisfied clause: body => true");
            return Ok(());
        }

        // If head is a constraint (not predicate, not false), transform:
        // body => constraint  -->  body ∧ NOT(constraint) => false
        if !is_head_predicate && !is_head_false {
            log::debug!(
                "Transforming non-predicate head: body => {} --> body ∧ NOT({}) => false",
                head,
                head
            );

            // Add NOT(head) to the body constraints
            let negated_head = ChcExpr::not(head.clone());
            let new_constraint = match constraint {
                Some(c) => Some(ChcExpr::and(c, negated_head)),
                None => Some(negated_head),
            };

            let clause_body = if new_constraint.is_some() || !preds.is_empty() {
                ClauseBody::new(preds, new_constraint)
            } else {
                ClauseBody::constraint(ChcExpr::Bool(true))
            };

            let clause = HornClause::new(clause_body, ClauseHead::False);
            self.problem.add_clause(clause);
        } else {
            // Normal case: head is a predicate or false
            // Apply relational encoding normalization: substitute primed variables
            // from body equalities into the head predicate
            let (normalized_constraint, substitutions) =
                self.normalize_relational_encoding(&preds, constraint, head);

            // Apply substitutions to head if any were found
            let normalized_head = if substitutions.is_empty() {
                head.clone()
            } else {
                self.apply_substitutions_to_head(head, &substitutions)
            };

            let clause_head = self.determine_head(&normalized_head)?;

            let clause_body = if normalized_constraint.is_some() || !preds.is_empty() {
                ClauseBody::new(preds, normalized_constraint)
            } else {
                ClauseBody::constraint(ChcExpr::Bool(true))
            };

            let clause = HornClause::new(clause_body, clause_head);
            self.problem.add_clause(clause);
        }

        Ok(())
    }

    /// Add a fact (head predicate that's always true)
    fn add_fact(&mut self, expr: ChcExpr) -> ChcResult<()> {
        let clause_head = self.determine_head(&expr)?;
        let clause = HornClause::new(ClauseBody::constraint(ChcExpr::Bool(true)), clause_head);
        self.problem.add_clause(clause);
        Ok(())
    }

    /// Extract predicate applications and constraints from body
    fn extract_body_parts(
        &self,
        body: &ChcExpr,
    ) -> (Vec<(PredicateId, Vec<ChcExpr>)>, Option<ChcExpr>) {
        let mut preds = Vec::new();
        let mut constraints = Vec::new();

        self.collect_body_parts(body, &mut preds, &mut constraints);

        let constraint = if constraints.is_empty() {
            None
        } else if constraints.len() == 1 {
            Some(constraints.into_iter().next().unwrap())
        } else {
            Some(constraints.into_iter().reduce(ChcExpr::and).unwrap())
        };

        (preds, constraint)
    }

    /// Recursively collect body parts
    fn collect_body_parts(
        &self,
        expr: &ChcExpr,
        preds: &mut Vec<(PredicateId, Vec<ChcExpr>)>,
        constraints: &mut Vec<ChcExpr>,
    ) {
        match expr {
            ChcExpr::Op(ChcOp::And, args) => {
                for arg in args {
                    self.collect_body_parts(arg, preds, constraints);
                }
            }
            _ => {
                // Check if this is a predicate application
                if let Some((pred_id, args)) = self.try_extract_predicate(expr) {
                    preds.push((pred_id, args));
                } else {
                    constraints.push(expr.clone());
                }
            }
        }
    }

    /// Try to extract a predicate application from an expression
    fn try_extract_predicate(&self, expr: &ChcExpr) -> Option<(PredicateId, Vec<ChcExpr>)> {
        match expr {
            ChcExpr::PredicateApp(_name, id, args) => {
                // Convert Arc<ChcExpr> to ChcExpr
                let arg_exprs: Vec<ChcExpr> = args.iter().map(|a| (**a).clone()).collect();
                Some((*id, arg_exprs))
            }
            _ => None,
        }
    }

    /// Determine the head of a clause from an expression
    fn determine_head(&self, expr: &ChcExpr) -> ChcResult<ClauseHead> {
        match expr {
            ChcExpr::Bool(false) => Ok(ClauseHead::False),
            ChcExpr::Bool(true) => {
                // true head means the clause is trivially satisfied - skip it
                Ok(ClauseHead::False) // This will be filtered out
            }
            ChcExpr::PredicateApp(_name, id, args) => {
                // Predicate application in the head
                let arg_exprs: Vec<ChcExpr> = args.iter().map(|a| (**a).clone()).collect();
                Ok(ClauseHead::Predicate(*id, arg_exprs))
            }
            _ => {
                // Other expressions - check if there's a nested predicate application
                if let Some((pred_id, args)) = self.try_extract_predicate(expr) {
                    Ok(ClauseHead::Predicate(pred_id, args))
                } else {
                    // Not a predicate - treat as a constraint that should be true
                    // This is typically an error in the input
                    log::warn!("Non-predicate expression in clause head: {}", expr);
                    Ok(ClauseHead::False)
                }
            }
        }
    }

    /// Normalize relational encoding by substituting primed variables in head predicates.
    ///
    /// For clauses with explicit primed variables like:
    ///   (Inv i) ∧ (< i 10) ∧ (= i_prime (+ i 1)) => Inv(i_prime)
    ///
    /// This transforms them to functional encoding:
    ///   (Inv i) ∧ (< i 10) => Inv((+ i 1))
    ///
    /// The transformation finds equalities `(= VAR EXPR)` or `(= EXPR VAR)` where:
    /// - VAR is a variable that appears in the head predicate arguments
    /// - VAR does NOT appear in any body predicate (it's a "next-state" variable)
    /// - VAR does NOT appear anywhere else in the constraint (except the defining equality)
    /// - EXPR is an expression defining VAR in terms of "current-state" variables
    ///
    /// It then substitutes VAR with EXPR in the head and removes the equality from the body.
    fn normalize_relational_encoding(
        &self,
        preds: &[(PredicateId, Vec<ChcExpr>)],
        constraint: Option<ChcExpr>,
        head: &ChcExpr,
    ) -> (Option<ChcExpr>, Vec<(ChcVar, ChcExpr)>) {
        // Only apply to clauses with body predicates (transition rules)
        // Fact clauses (no body predicates) should not be transformed
        if preds.is_empty() {
            return (constraint, Vec::new());
        }

        // Collect variables that appear in the head predicate arguments
        let head_vars: FxHashSet<ChcVar> = head.vars().into_iter().collect();

        if head_vars.is_empty() {
            return (constraint, Vec::new());
        }

        // Collect variables that appear in body predicates ("current-state" variables)
        // These should NOT be substituted, only "next-state" variables should be
        let mut body_pred_vars: FxHashSet<ChcVar> = FxHashSet::default();
        for (_pred_id, args) in preds {
            for arg in args {
                for v in arg.vars() {
                    body_pred_vars.insert(v);
                }
            }
        }

        // Extract equalities from the constraint
        let Some(ref cstr) = constraint else {
            return (constraint, Vec::new());
        };

        let mut substitutions: Vec<(ChcVar, ChcExpr)> = Vec::new();
        let mut equalities_to_remove: FxHashSet<usize> = FxHashSet::default();

        // Collect all conjuncts from the constraint
        let conjuncts = self.collect_conjuncts(cstr);

        // Build a set of variables used in each conjunct (for checking if var is used elsewhere)
        let conjunct_vars: Vec<FxHashSet<ChcVar>> = conjuncts
            .iter()
            .map(|c| c.vars().into_iter().collect())
            .collect();

        for (idx, conjunct) in conjuncts.iter().enumerate() {
            if let Some((var, expr)) = self.extract_var_expr_equality(conjunct) {
                // Check if this variable appears in the head
                if !head_vars.contains(&var) {
                    continue;
                }

                // CRITICAL: Only substitute if the variable does NOT appear in any body predicate
                // This distinguishes relational encoding (i_prime only in head + equality)
                // from constraint encoding (x in body predicate + equality like x = 0)
                if body_pred_vars.contains(&var) {
                    continue;
                }

                // CRITICAL: Check if the variable appears in any OTHER conjunct (not just body preds)
                // This handles fact clauses where constraints like (= x 0) define values
                let var_used_elsewhere = conjunct_vars
                    .iter()
                    .enumerate()
                    .any(|(i, vars)| i != idx && vars.contains(&var));

                if var_used_elsewhere {
                    continue;
                }

                // Check that expr variables are all from body predicates (current-state)
                // This ensures we're substituting next-state = f(current-state)
                let expr_vars: FxHashSet<ChcVar> = expr.vars().into_iter().collect();

                // Accept substitution if all variables in the expression appear in body predicates
                // (we already checked preds is non-empty, so body_pred_vars should be populated)
                let expr_uses_current_state = expr_vars.iter().all(|v| body_pred_vars.contains(v));

                if expr_uses_current_state {
                    log::debug!(
                        "Relational encoding: substituting {} with {} in head",
                        var,
                        expr
                    );
                    substitutions.push((var, expr));
                    equalities_to_remove.insert(idx);
                }
            }
        }

        if substitutions.is_empty() {
            return (constraint, Vec::new());
        }

        // Rebuild constraint without the removed equalities
        let remaining_conjuncts: Vec<ChcExpr> = conjuncts
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| !equalities_to_remove.contains(idx))
            .map(|(_, c)| c)
            .collect();

        let new_constraint = if remaining_conjuncts.is_empty() {
            None
        } else if remaining_conjuncts.len() == 1 {
            Some(remaining_conjuncts.into_iter().next().unwrap())
        } else {
            Some(
                remaining_conjuncts
                    .into_iter()
                    .reduce(ChcExpr::and)
                    .unwrap(),
            )
        };

        (new_constraint, substitutions)
    }

    /// Collect all conjuncts from an AND expression (flattened)
    fn collect_conjuncts(&self, expr: &ChcExpr) -> Vec<ChcExpr> {
        let mut result = Vec::new();
        self.collect_conjuncts_recursive(expr, &mut result);
        result
    }

    #[allow(clippy::only_used_in_recursion)]
    fn collect_conjuncts_recursive(&self, expr: &ChcExpr, result: &mut Vec<ChcExpr>) {
        match expr {
            ChcExpr::Op(ChcOp::And, args) => {
                for arg in args {
                    self.collect_conjuncts_recursive(arg, result);
                }
            }
            _ => {
                result.push(expr.clone());
            }
        }
    }

    /// Extract (variable, expression) pair from an equality constraint.
    /// Returns Some((var, expr)) if the equality is of form (= VAR EXPR) or (= EXPR VAR)
    /// where VAR is a simple variable and EXPR is not the same variable.
    fn extract_var_expr_equality(&self, expr: &ChcExpr) -> Option<(ChcVar, ChcExpr)> {
        if let ChcExpr::Op(ChcOp::Eq, args) = expr {
            if args.len() == 2 {
                let left = args[0].as_ref();
                let right = args[1].as_ref();

                // Check (= VAR EXPR) pattern
                if let ChcExpr::Var(v) = left {
                    // Make sure right side is not the same variable
                    if !matches!(right, ChcExpr::Var(v2) if v2 == v) {
                        return Some((v.clone(), right.clone()));
                    }
                }

                // Check (= EXPR VAR) pattern
                if let ChcExpr::Var(v) = right {
                    // Make sure left side is not the same variable
                    if !matches!(left, ChcExpr::Var(v2) if v2 == v) {
                        return Some((v.clone(), left.clone()));
                    }
                }
            }
        }
        None
    }

    /// Apply substitutions to a head expression (predicate application)
    fn apply_substitutions_to_head(&self, head: &ChcExpr, subst: &[(ChcVar, ChcExpr)]) -> ChcExpr {
        if subst.is_empty() {
            return head.clone();
        }
        head.substitute(subst)
    }

    /// Parse a sort
    fn parse_sort(&mut self) -> ChcResult<ChcSort> {
        self.skip_whitespace_and_comments();

        if self.peek_char() == Some('(') {
            // Compound sort: (_ BitVec 32) or (Array Int Int)
            self.expect_char('(')?;
            self.skip_whitespace_and_comments();

            // Check if it's indexed (_ ...) or parametric (Array ...)
            let first = self.parse_symbol()?;
            self.skip_whitespace_and_comments();

            match first.as_str() {
                "_" => {
                    // Indexed sort: (_ BitVec 32)
                    let sort_name = self.parse_symbol()?;
                    self.skip_whitespace_and_comments();

                    match sort_name.as_str() {
                        "BitVec" => {
                            let width: u32 = self
                                .parse_numeral()?
                                .parse()
                                .map_err(|_| ChcError::Parse("Invalid bitvector width".into()))?;
                            self.skip_whitespace_and_comments();
                            self.expect_char(')')?;
                            Ok(ChcSort::BitVec(width))
                        }
                        _ => Err(ChcError::Parse(format!(
                            "Unknown indexed sort: {}",
                            sort_name
                        ))),
                    }
                }
                "Array" => {
                    // Parametric sort: (Array key_sort value_sort)
                    let key_sort = self.parse_sort()?;
                    self.skip_whitespace_and_comments();
                    let value_sort = self.parse_sort()?;
                    self.skip_whitespace_and_comments();
                    self.expect_char(')')?;
                    Ok(ChcSort::Array(Box::new(key_sort), Box::new(value_sort)))
                }
                _ => Err(ChcError::Parse(format!(
                    "Unknown parametric sort: {}",
                    first
                ))),
            }
        } else {
            let name = self.parse_symbol()?;
            match name.as_str() {
                "Bool" => Ok(ChcSort::Bool),
                "Int" => Ok(ChcSort::Int),
                "Real" => Ok(ChcSort::Real),
                _ => Err(ChcError::Parse(format!("Unknown sort: {}", name))),
            }
        }
    }

    /// Parse an expression
    fn parse_expr(&mut self) -> ChcResult<ChcExpr> {
        self.skip_whitespace_and_comments();

        match self.peek_char() {
            Some('(') => self.parse_compound_expr(),
            Some(c) if c.is_ascii_digit() || c == '-' => self.parse_numeral_expr(),
            Some(_) => self.parse_symbol_expr(),
            None => Err(ChcError::Parse("Unexpected end of input".into())),
        }
    }

    /// Parse a compound expression (function application)
    fn parse_compound_expr(&mut self) -> ChcResult<ChcExpr> {
        self.expect_char('(')?;
        self.skip_whitespace_and_comments();

        // Check for special forms
        if self.peek_char() == Some('_') {
            // Indexed identifier
            return self.parse_indexed_expr();
        }

        // Check for let
        let first = self.parse_symbol()?;
        self.skip_whitespace_and_comments();

        match first.as_str() {
            "let" => self.parse_let_expr(),
            "forall" | "exists" => self.parse_quantifier_expr(&first),
            _ => self.parse_application(&first),
        }
    }

    /// Parse indexed expression like (_ bv123 32)
    fn parse_indexed_expr(&mut self) -> ChcResult<ChcExpr> {
        self.expect_char('_')?;
        self.skip_whitespace_and_comments();

        let name = self.parse_symbol()?;
        self.skip_whitespace_and_comments();

        // Parse remaining arguments
        let mut args = Vec::new();
        while self.peek_char() != Some(')') {
            args.push(self.parse_symbol()?);
            self.skip_whitespace_and_comments();
        }
        self.expect_char(')')?;

        match name.as_str() {
            _ if name.starts_with("bv") => {
                // Bitvector literal
                let value: u64 = name[2..]
                    .parse()
                    .map_err(|_| ChcError::Parse(format!("Invalid bitvector literal: {}", name)))?;
                let _width: u32 = args
                    .first()
                    .ok_or_else(|| ChcError::Parse("Missing bitvector width".into()))?
                    .parse()
                    .map_err(|_| ChcError::Parse("Invalid bitvector width".into()))?;
                Ok(ChcExpr::Int(value as i64))
            }
            _ => Err(ChcError::Parse(format!(
                "Unknown indexed identifier: {}",
                name
            ))),
        }
    }

    /// Parse let expression
    fn parse_let_expr(&mut self) -> ChcResult<ChcExpr> {
        self.skip_whitespace_and_comments();
        self.expect_char('(')?;

        // Parse bindings
        let mut bindings = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            if self.peek_char() == Some(')') {
                break;
            }
            self.expect_char('(')?;
            self.skip_whitespace_and_comments();
            let var_name = self.parse_symbol()?;
            self.skip_whitespace_and_comments();
            let value = self.parse_expr()?;
            self.skip_whitespace_and_comments();
            self.expect_char(')')?;
            bindings.push((var_name, value));
        }
        self.expect_char(')')?;

        // Add let-bound variables to the variable map before parsing body
        // This ensures that references to these variables get the correct sort
        let mut old_values = Vec::new();
        for (name, value) in &bindings {
            let sort = value.sort();
            let old = self.variables.insert(name.clone(), sort);
            old_values.push((name.clone(), old));
        }

        self.skip_whitespace_and_comments();
        let body = self.parse_expr()?;
        self.skip_whitespace_and_comments();
        self.expect_char(')')?;

        // Restore original variable bindings
        for (name, old) in old_values {
            match old {
                Some(sort) => {
                    self.variables.insert(name, sort);
                }
                None => {
                    self.variables.remove(&name);
                }
            }
        }

        // Substitute bindings in body
        let substitutions: Vec<(ChcVar, ChcExpr)> = bindings
            .into_iter()
            .map(|(name, value)| {
                let sort = value.sort();
                (ChcVar::new(name, sort), value)
            })
            .collect();

        Ok(body.substitute(&substitutions))
    }

    /// Parse quantifier expression (forall/exists)
    fn parse_quantifier_expr(&mut self, _quantifier: &str) -> ChcResult<ChcExpr> {
        self.skip_whitespace_and_comments();
        self.expect_char('(')?;

        // Parse variable declarations
        let mut vars = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            if self.peek_char() == Some(')') {
                break;
            }
            self.expect_char('(')?;
            self.skip_whitespace_and_comments();
            let var_name = self.parse_symbol()?;
            self.skip_whitespace_and_comments();
            let sort = self.parse_sort()?;
            self.skip_whitespace_and_comments();
            self.expect_char(')')?;

            // Register variable
            self.variables.insert(var_name.clone(), sort.clone());
            vars.push(ChcVar::new(var_name, sort));
        }
        self.expect_char(')')?;

        self.skip_whitespace_and_comments();
        let body = self.parse_expr()?;
        self.skip_whitespace_and_comments();
        self.expect_char(')')?;

        // For CHC, we treat forall as implicit and return the body
        // The variables are already registered
        Ok(body)
    }

    /// Parse function application
    fn parse_application(&mut self, func: &str) -> ChcResult<ChcExpr> {
        let mut args = Vec::new();

        loop {
            self.skip_whitespace_and_comments();
            if self.peek_char() == Some(')') {
                break;
            }
            args.push(self.parse_expr()?);
        }
        self.expect_char(')')?;

        // Map function names to operations
        match func {
            "not" => {
                if args.len() != 1 {
                    return Err(ChcError::Parse("'not' requires exactly 1 argument".into()));
                }
                Ok(ChcExpr::not(args.into_iter().next().unwrap()))
            }
            "and" => {
                if args.is_empty() {
                    Ok(ChcExpr::Bool(true))
                } else {
                    Ok(args.into_iter().reduce(ChcExpr::and).unwrap())
                }
            }
            "or" => {
                if args.is_empty() {
                    Ok(ChcExpr::Bool(false))
                } else {
                    Ok(args.into_iter().reduce(ChcExpr::or).unwrap())
                }
            }
            "=>" | "implies" => {
                if args.len() != 2 {
                    return Err(ChcError::Parse("'=>' requires exactly 2 arguments".into()));
                }
                let mut iter = args.into_iter();
                Ok(ChcExpr::implies(iter.next().unwrap(), iter.next().unwrap()))
            }
            "=" => {
                if args.len() != 2 {
                    return Err(ChcError::Parse("'=' requires exactly 2 arguments".into()));
                }
                let mut iter = args.into_iter();
                Ok(ChcExpr::eq(iter.next().unwrap(), iter.next().unwrap()))
            }
            "distinct" => {
                if args.len() != 2 {
                    return Err(ChcError::Parse(
                        "'distinct' requires exactly 2 arguments".into(),
                    ));
                }
                let mut iter = args.into_iter();
                Ok(ChcExpr::ne(iter.next().unwrap(), iter.next().unwrap()))
            }
            "<" => {
                if args.len() != 2 {
                    return Err(ChcError::Parse("'<' requires exactly 2 arguments".into()));
                }
                let mut iter = args.into_iter();
                Ok(ChcExpr::lt(iter.next().unwrap(), iter.next().unwrap()))
            }
            "<=" => {
                if args.len() != 2 {
                    return Err(ChcError::Parse("'<=' requires exactly 2 arguments".into()));
                }
                let mut iter = args.into_iter();
                Ok(ChcExpr::le(iter.next().unwrap(), iter.next().unwrap()))
            }
            ">" => {
                if args.len() != 2 {
                    return Err(ChcError::Parse("'>' requires exactly 2 arguments".into()));
                }
                let mut iter = args.into_iter();
                Ok(ChcExpr::gt(iter.next().unwrap(), iter.next().unwrap()))
            }
            ">=" => {
                if args.len() != 2 {
                    return Err(ChcError::Parse("'>=' requires exactly 2 arguments".into()));
                }
                let mut iter = args.into_iter();
                Ok(ChcExpr::ge(iter.next().unwrap(), iter.next().unwrap()))
            }
            "+" => {
                if args.is_empty() {
                    Ok(ChcExpr::int(0))
                } else {
                    Ok(args.into_iter().reduce(ChcExpr::add).unwrap())
                }
            }
            "-" => {
                if args.is_empty() {
                    return Err(ChcError::Parse("'-' requires at least 1 argument".into()));
                }
                if args.len() == 1 {
                    Ok(ChcExpr::neg(args.into_iter().next().unwrap()))
                } else {
                    Ok(args.into_iter().reduce(ChcExpr::sub).unwrap())
                }
            }
            "*" => {
                if args.is_empty() {
                    Ok(ChcExpr::int(1))
                } else {
                    Ok(args.into_iter().reduce(ChcExpr::mul).unwrap())
                }
            }
            "div" => {
                if args.len() != 2 {
                    return Err(ChcError::Parse("'div' requires exactly 2 arguments".into()));
                }
                let mut iter = args.into_iter();
                let a = iter.next().unwrap();
                let b = iter.next().unwrap();
                Ok(ChcExpr::Op(ChcOp::Div, vec![Arc::new(a), Arc::new(b)]))
            }
            "mod" => {
                if args.len() != 2 {
                    return Err(ChcError::Parse("'mod' requires exactly 2 arguments".into()));
                }
                let mut iter = args.into_iter();
                let a = iter.next().unwrap();
                let b = iter.next().unwrap();
                Ok(ChcExpr::Op(ChcOp::Mod, vec![Arc::new(a), Arc::new(b)]))
            }
            "ite" => {
                if args.len() != 3 {
                    return Err(ChcError::Parse("'ite' requires exactly 3 arguments".into()));
                }
                let mut iter = args.into_iter();
                Ok(ChcExpr::ite(
                    iter.next().unwrap(),
                    iter.next().unwrap(),
                    iter.next().unwrap(),
                ))
            }
            "select" => {
                if args.len() != 2 {
                    return Err(ChcError::Parse(
                        "'select' requires exactly 2 arguments".into(),
                    ));
                }
                let mut iter = args.into_iter();
                Ok(ChcExpr::select(iter.next().unwrap(), iter.next().unwrap()))
            }
            "store" => {
                if args.len() != 3 {
                    return Err(ChcError::Parse(
                        "'store' requires exactly 3 arguments".into(),
                    ));
                }
                let mut iter = args.into_iter();
                Ok(ChcExpr::store(
                    iter.next().unwrap(),
                    iter.next().unwrap(),
                    iter.next().unwrap(),
                ))
            }
            "true" => Ok(ChcExpr::Bool(true)),
            "false" => Ok(ChcExpr::Bool(false)),
            _ => {
                // Check if it's a predicate application
                if let Some((pred_id, _sorts)) = self.predicates.get(func).cloned() {
                    // It's a predicate application - create PredicateApp expression
                    Ok(ChcExpr::predicate_app(func, pred_id, args))
                } else {
                    // Unknown function - treat as uninterpreted (log warning)
                    log::warn!("Unknown function: {}", func);
                    Ok(ChcExpr::Bool(true)) // Placeholder for unknown functions
                }
            }
        }
    }

    /// Parse a numeral expression
    fn parse_numeral_expr(&mut self) -> ChcResult<ChcExpr> {
        let num_str = self.parse_numeral()?;
        let n: i64 = num_str
            .parse()
            .map_err(|_| ChcError::Parse(format!("Invalid numeral: {}", num_str)))?;
        Ok(ChcExpr::int(n))
    }

    /// Parse a symbol expression (variable or constant)
    fn parse_symbol_expr(&mut self) -> ChcResult<ChcExpr> {
        let name = self.parse_symbol()?;

        match name.as_str() {
            "true" => Ok(ChcExpr::Bool(true)),
            "false" => Ok(ChcExpr::Bool(false)),
            _ => {
                // Check if it's a nullary predicate application first
                if let Some((pred_id, sorts)) = self.predicates.get(&name).cloned() {
                    if sorts.is_empty() {
                        // Nullary predicate - create a PredicateApp with no arguments
                        return Ok(ChcExpr::predicate_app(&name, pred_id, Vec::new()));
                    }
                }
                // Look up variable
                if let Some(sort) = self.variables.get(&name).cloned() {
                    Ok(ChcExpr::var(ChcVar::new(name, sort)))
                } else {
                    // Assume it's an integer variable if not declared
                    Ok(ChcExpr::var(ChcVar::new(name, ChcSort::Int)))
                }
            }
        }
    }

    /// Parse a symbol
    fn parse_symbol(&mut self) -> ChcResult<String> {
        self.skip_whitespace_and_comments();

        let start = self.pos;

        // Check for quoted symbol
        if self.peek_char() == Some('|') {
            self.pos += 1;
            let content_start = self.pos;
            while self.pos < self.input.len() && self.current_char() != Some('|') {
                self.pos += 1;
            }
            let symbol = self.input[content_start..self.pos].to_string();
            if self.current_char() == Some('|') {
                self.pos += 1;
            }
            return Ok(symbol);
        }

        // Regular symbol
        while self.pos < self.input.len() {
            match self.current_char() {
                Some(c) if is_symbol_char(c) => self.pos += 1,
                _ => break,
            }
        }

        if start == self.pos {
            return Err(ChcError::Parse("Expected symbol".into()));
        }

        Ok(self.input[start..self.pos].to_string())
    }

    /// Parse a numeral
    fn parse_numeral(&mut self) -> ChcResult<String> {
        self.skip_whitespace_and_comments();

        let start = self.pos;
        let mut has_sign = false;

        // Optional sign
        if self.peek_char() == Some('-') {
            self.pos += 1;
            has_sign = true;
        }

        // Digits
        while self.pos < self.input.len() {
            match self.current_char() {
                Some(c) if c.is_ascii_digit() => self.pos += 1,
                _ => break,
            }
        }

        if start == self.pos || (has_sign && self.pos == start + 1) {
            return Err(ChcError::Parse("Expected numeral".into()));
        }

        Ok(self.input[start..self.pos].to_string())
    }

    /// Skip whitespace and comments
    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.input.len() {
            match self.current_char() {
                Some(c) if c.is_whitespace() => self.pos += c.len_utf8(),
                Some(';') => {
                    // Skip until end of line (handle multi-byte UTF-8 chars in comments)
                    while self.pos < self.input.len() {
                        if let Some(c) = self.current_char() {
                            if c == '\n' {
                                break;
                            }
                            self.pos += c.len_utf8();
                        } else {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
    }

    /// Expect and consume a specific character
    fn expect_char(&mut self, expected: char) -> ChcResult<()> {
        self.skip_whitespace_and_comments();
        match self.current_char() {
            Some(c) if c == expected => {
                self.pos += 1;
                Ok(())
            }
            Some(c) => Err(ChcError::Parse(format!(
                "Expected '{}', found '{}'",
                expected, c
            ))),
            None => Err(ChcError::Parse(format!(
                "Expected '{}', found end of input",
                expected
            ))),
        }
    }

    /// Get current character
    fn current_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    /// Peek at current character without consuming
    fn peek_char(&self) -> Option<char> {
        self.current_char()
    }
}

impl Default for ChcParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a character is valid in a symbol
/// Note: includes `'` for primed variables (e.g., x' for next-state)
fn is_symbol_char(c: char) -> bool {
    c.is_alphanumeric()
        || matches!(
            c,
            '_' | '-'
                | '+'
                | '*'
                | '/'
                | '.'
                | '!'
                | '@'
                | '#'
                | '$'
                | '%'
                | '^'
                | '&'
                | '<'
                | '>'
                | '='
                | '?'
                | '~'
                | '\''
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_chc() {
        let input = r#"
            (set-logic HORN)
            (declare-rel Inv (Int))
            (declare-var x Int)
        "#;

        let problem = ChcParser::parse(input).unwrap();
        assert_eq!(problem.predicates().len(), 1);
    }

    #[test]
    fn test_parse_sort() {
        let mut parser = ChcParser::new();
        parser.input = "Int".to_string();
        parser.pos = 0;
        let sort = parser.parse_sort().unwrap();
        assert_eq!(sort, ChcSort::Int);

        parser.input = "Bool".to_string();
        parser.pos = 0;
        let sort = parser.parse_sort().unwrap();
        assert_eq!(sort, ChcSort::Bool);

        parser.input = "(_ BitVec 32)".to_string();
        parser.pos = 0;
        let sort = parser.parse_sort().unwrap();
        assert_eq!(sort, ChcSort::BitVec(32));
    }

    #[test]
    fn test_parse_expr_literal() {
        let mut parser = ChcParser::new();

        parser.input = "42".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        assert_eq!(expr, ChcExpr::int(42));

        parser.input = "-10".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        assert_eq!(expr, ChcExpr::int(-10));

        parser.input = "true".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        assert_eq!(expr, ChcExpr::Bool(true));
    }

    #[test]
    fn test_parse_expr_arithmetic() {
        let mut parser = ChcParser::new();
        parser.variables.insert("x".to_string(), ChcSort::Int);

        parser.input = "(+ x 1)".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        // Just check it parses without error
        assert!(matches!(expr, ChcExpr::Op(ChcOp::Add, _)));

        parser.input = "(- x 5)".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        assert!(matches!(expr, ChcExpr::Op(ChcOp::Sub, _)));
    }

    #[test]
    fn test_parse_expr_comparison() {
        let mut parser = ChcParser::new();
        parser.variables.insert("x".to_string(), ChcSort::Int);

        parser.input = "(< x 10)".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        assert!(matches!(expr, ChcExpr::Op(ChcOp::Lt, _)));

        parser.input = "(= x 0)".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        assert!(matches!(expr, ChcExpr::Op(ChcOp::Eq, _)));
    }

    #[test]
    fn test_parse_expr_boolean() {
        let mut parser = ChcParser::new();
        parser.variables.insert("a".to_string(), ChcSort::Bool);
        parser.variables.insert("b".to_string(), ChcSort::Bool);

        parser.input = "(and a b)".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        assert!(matches!(expr, ChcExpr::Op(ChcOp::And, _)));

        parser.input = "(or a b)".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        assert!(matches!(expr, ChcExpr::Op(ChcOp::Or, _)));

        parser.input = "(not a)".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        assert!(matches!(expr, ChcExpr::Op(ChcOp::Not, _)));
    }

    #[test]
    fn test_parse_expr_implication() {
        let mut parser = ChcParser::new();
        parser.variables.insert("x".to_string(), ChcSort::Int);

        parser.input = "(=> (= x 0) (< x 10))".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        assert!(matches!(expr, ChcExpr::Op(ChcOp::Implies, _)));
    }

    #[test]
    fn test_parse_let_expr() {
        let mut parser = ChcParser::new();

        parser.input = "(let ((x 5)) (+ x 1))".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        // After substitution, should be (+ 5 1)
        assert!(matches!(expr, ChcExpr::Op(ChcOp::Add, _)));
    }

    #[test]
    fn test_parse_forall_expr() {
        let mut parser = ChcParser::new();

        parser.input = "(forall ((x Int)) (>= x 0))".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        // Forall is stripped for CHC, just returns body
        assert!(matches!(expr, ChcExpr::Op(ChcOp::Ge, _)));
    }

    #[test]
    fn test_parse_chc_with_rule() {
        let input = r#"
            (set-logic HORN)
            (declare-rel Inv (Int))
            (declare-var x Int)
            (rule (=> (= x 0) (Inv x)))
        "#;

        let problem = ChcParser::parse(input).unwrap();
        assert_eq!(problem.predicates().len(), 1);
        // Rule should be added
        assert!(!problem.clauses().is_empty());
    }

    #[test]
    fn test_parse_comments() {
        let input = r#"
            ; This is a comment
            (set-logic HORN) ; inline comment
            (declare-rel Inv (Int)) ; another comment
        "#;

        let problem = ChcParser::parse(input).unwrap();
        assert_eq!(problem.predicates().len(), 1);
    }

    #[test]
    fn test_parse_predicate_application() {
        let input = r#"
            (set-logic HORN)
            (declare-rel Inv (Int))
            (declare-var x Int)
            (rule (=> (= x 0) (Inv x)))
        "#;

        let problem = ChcParser::parse(input).unwrap();
        assert_eq!(problem.predicates().len(), 1);
        assert_eq!(problem.clauses().len(), 1);

        // Check that the clause has a predicate head
        let clause = &problem.clauses()[0];
        assert!(matches!(clause.head, ClauseHead::Predicate(_, _)));
    }

    #[test]
    fn test_parse_predicate_in_body() {
        let input = r#"
            (set-logic HORN)
            (declare-rel Inv (Int))
            (declare-var x Int)
            (rule (=> (= x 0) (Inv x)))
            (rule (=> (and (Inv x) (< x 10)) (Inv (+ x 1))))
        "#;

        let problem = ChcParser::parse(input).unwrap();
        assert_eq!(problem.clauses().len(), 2);

        // Second clause should have a predicate in the body
        let clause = &problem.clauses()[1];
        assert!(
            !clause.body.predicates.is_empty(),
            "Body should contain predicate application"
        );
    }

    #[test]
    fn test_parse_query() {
        let input = r#"
            (set-logic HORN)
            (declare-rel Inv (Int))
            (declare-var x Int)
            (rule (=> (= x 0) (Inv x)))
            (query Inv)
        "#;

        let problem = ChcParser::parse(input).unwrap();
        // Should have 2 clauses: 1 rule + 1 query
        assert_eq!(problem.clauses().len(), 2);

        // Query should have False head
        let query_clause = &problem.clauses()[1];
        assert!(query_clause.is_query());
    }

    #[test]
    fn test_parse_primed_variables() {
        let mut parser = ChcParser::new();
        parser.variables.insert("x".to_string(), ChcSort::Int);
        parser.variables.insert("x'".to_string(), ChcSort::Int);

        parser.input = "x'".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();

        match expr {
            ChcExpr::Var(v) => {
                assert!(v.is_primed());
                assert_eq!(v.base_name(), "x");
            }
            _ => panic!("Expected variable expression"),
        }
    }

    #[test]
    fn test_parse_counter_safe_example() {
        let input = r#"
            (set-logic HORN)
            (declare-rel Inv (Int))
            (declare-var x Int)
            (rule (=> (= x 0) (Inv x)))
            (rule (=> (and (Inv x) (< x 10)) (Inv (+ x 1))))
            (query (and (Inv x) (> x 10)))
        "#;

        let problem = ChcParser::parse(input).unwrap();
        assert_eq!(problem.predicates().len(), 1);
        assert_eq!(problem.clauses().len(), 3); // 2 rules + 1 query

        // Last clause should be a query
        let query = &problem.clauses()[2];
        assert!(query.is_query());
    }

    #[test]
    fn test_parse_multi_predicate() {
        let input = r#"
            (set-logic HORN)
            (declare-rel P (Int))
            (declare-rel Q (Int Int))
            (declare-var x Int)
            (declare-var y Int)
            (rule (=> (= x 0) (P x)))
            (rule (=> (and (P x) (= y (+ x 1))) (Q x y)))
        "#;

        let problem = ChcParser::parse(input).unwrap();
        assert_eq!(problem.predicates().len(), 2);
        assert_eq!(problem.clauses().len(), 2);
    }

    #[test]
    fn test_parse_nested_let_expr() {
        let mut parser = ChcParser::new();

        // Test nested let bindings like in three_dots_moving_2
        parser.input = "(let ((a!1 5)) (let ((a!2 (+ a!1 1))) (+ a!2 1)))".to_string();
        parser.pos = 0;
        let expr = parser.parse_expr().unwrap();
        // After substitution of a!1=5, then a!2=(+ 5 1), should be (+ (+ 5 1) 1)
        // (No constant folding, but substitution works correctly)
        assert!(matches!(expr, ChcExpr::Op(ChcOp::Add, _)));
    }

    #[test]
    fn test_parse_three_dots_clause() {
        // This is the problematic clause from three_dots_moving_2_000.smt2
        let input = r#"(set-logic HORN)
(declare-fun inv (Int Int Int Int) Bool)
(assert
  (forall ((A Int) (B Int) (C Int) (D Int) (E Int) (F Int) (G Int))
    (=>
      (and
        (inv B C F A)
        (let ((a!1 (and (= D (ite (<= B F) (+ 1 B) (+ (- 1) B))) (= E D) (= B C))))
        (let ((a!2 (or (and (= D B) (= E (+ (- 1) C)) (not (= B C))) a!1)))
          (and (= G (+ (- 1) A)) a!2 (not (= C F))))))
      (inv D E F G))))
(check-sat)"#;

        let problem = ChcParser::parse(input).expect("parse failed");

        // Should have 1 clause (the transition rule)
        assert_eq!(problem.clauses().len(), 1);

        let clause = &problem.clauses()[0];

        // Check the constraint doesn't contain raw "a!1" or "a!2" variables
        if let Some(ref constraint) = clause.body.constraint {
            let constraint_str = format!("{}", constraint);
            println!("Constraint: {}", constraint_str);

            // The let bindings should be expanded - no a!1 or a!2 should appear
            assert!(
                !constraint_str.contains("a!1"),
                "Found unexpanded a!1 in constraint: {}",
                constraint_str
            );
            assert!(
                !constraint_str.contains("a!2"),
                "Found unexpanded a!2 in constraint: {}",
                constraint_str
            );
        }
    }

    #[test]
    fn test_relational_encoding_normalization() {
        // Test relational encoding: (Inv i) /\ (< i 10) /\ (= i' (+ i 1)) => Inv(i')
        // Should be normalized to: (Inv i) /\ (< i 10) => Inv((+ i 1))
        let input = r#"
            (set-logic HORN)
            (declare-fun Inv (Int) Bool)
            (assert (forall ((i Int)) (=> (= i 0) (Inv i))))
            (assert (forall ((i Int) (i_prime Int))
              (=> (and (Inv i) (< i 10) (= i_prime (+ i 1))) (Inv i_prime))))
            (check-sat)
        "#;

        let problem = ChcParser::parse(input).unwrap();
        assert_eq!(problem.predicates().len(), 1);
        assert_eq!(problem.clauses().len(), 2);

        // First clause: fact (= i 0) => Inv(i)
        // This should NOT be transformed (no body predicate)
        let init_clause = &problem.clauses()[0];
        assert!(init_clause.body.predicates.is_empty());
        // The head should still have a variable argument, not a constant
        if let crate::ClauseHead::Predicate(_, args) = &init_clause.head {
            assert_eq!(args.len(), 1);
            // The argument should be a variable (i), not a constant (0)
            assert!(
                matches!(args[0], ChcExpr::Var(_)),
                "Init clause head should have variable arg"
            );
        }

        // Second clause: transition rule should be normalized
        // i_prime should be replaced with (+ i 1) in the head
        let trans_clause = &problem.clauses()[1];
        assert!(!trans_clause.body.predicates.is_empty()); // Has body predicate
        if let crate::ClauseHead::Predicate(_, args) = &trans_clause.head {
            assert_eq!(args.len(), 1);
            // The argument should be an expression (+ i 1), not a variable (i_prime)
            let head_arg = &args[0];
            assert!(
                matches!(head_arg, ChcExpr::Op(ChcOp::Add, _)),
                "Trans clause head should have (+ i 1) expression, got: {}",
                head_arg
            );
        }

        // The constraint should NOT contain (= i_prime ...) anymore
        if let Some(ref constraint) = trans_clause.body.constraint {
            let constraint_str = format!("{}", constraint);
            assert!(
                !constraint_str.contains("i_prime"),
                "i_prime equality should be removed from constraint: {}",
                constraint_str
            );
        }
    }

    #[test]
    fn test_relational_encoding_preserves_facts() {
        // Fact clauses should NOT be transformed
        // (= x 0) /\ (= y 0) => Inv(x, y) should keep x and y as variables in head
        let input = r#"
            (set-logic HORN)
            (declare-fun Inv (Int Int) Bool)
            (assert (forall ((x Int) (y Int))
              (=> (and (= x 0) (= y 0)) (Inv x y))))
            (check-sat)
        "#;

        let problem = ChcParser::parse(input).unwrap();
        assert_eq!(problem.clauses().len(), 1);

        let clause = &problem.clauses()[0];
        // This is a fact (no body predicates)
        assert!(clause.body.predicates.is_empty());

        // Head should have variables, not constants
        if let crate::ClauseHead::Predicate(_, args) = &clause.head {
            assert_eq!(args.len(), 2);
            assert!(
                matches!(args[0], ChcExpr::Var(_)),
                "First arg should be variable"
            );
            assert!(
                matches!(args[1], ChcExpr::Var(_)),
                "Second arg should be variable"
            );
        }

        // Constraint should still contain the equalities
        if let Some(ref constraint) = clause.body.constraint {
            let constraint_str = format!("{}", constraint);
            assert!(
                constraint_str.contains("= x 0") || constraint_str.contains("(= 0 x)"),
                "Constraint should contain x = 0: {}",
                constraint_str
            );
        }
    }
}
