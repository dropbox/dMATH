//! Loop invariant inference engine
//!
//! This module provides automatic inference of loop invariants for simple
//! patterns commonly found in C code. The inference is based on:
//!
//! 1. **Variable bounds**: Track variables that are bounded by constants
//! 2. **Monotonicity**: Track variables that monotonically increase/decrease
//! 3. **Preservation**: Check what properties are preserved by the loop body
//!
//! ## Supported Patterns
//!
//! - **Counter loops**: `for (i = 0; i < n; i++)` infers `0 <= i <= n`
//! - **Accumulator loops**: Sum/product patterns
//! - **Search loops**: Loops that search for a condition

use std::collections::HashMap;

use crate::expr::{BinOp, CExpr, UnaryOp};
use crate::spec::Spec;
use crate::stmt::CStmt;

/// Loop invariant inference engine
pub struct InvariantInference {
    /// Variables that increase each iteration
    increasing_vars: Vec<String>,
    /// Variables that decrease each iteration
    decreasing_vars: Vec<String>,
    /// Known bounds on variables (var, lower, upper)
    bounds: Vec<(String, Option<i64>, Option<i64>)>,
}

impl Default for InvariantInference {
    fn default() -> Self {
        Self::new()
    }
}

impl InvariantInference {
    pub fn new() -> Self {
        Self {
            increasing_vars: Vec::new(),
            decreasing_vars: Vec::new(),
            bounds: Vec::new(),
        }
    }

    /// Infer a loop invariant from a while loop
    ///
    /// Returns a list of candidate invariants, ordered by likelihood
    pub fn infer_while_invariant(
        &mut self,
        cond: &CExpr,
        body: &CStmt,
        context: &InferenceContext,
    ) -> Vec<Spec> {
        let mut invariants = Vec::new();

        // Analyze the loop condition
        self.analyze_condition(cond);

        // Analyze the loop body for variable modifications
        self.analyze_body(body);

        // 1. Generate bound invariants for counter variables
        for (var, lower, upper) in &self.bounds {
            if let Some(lo) = lower {
                invariants.push(Spec::ge(Spec::var(var), Spec::int(*lo)));
            }
            if let Some(hi) = upper {
                // For < condition, use <= hi - 1 or < hi depending on context
                invariants.push(Spec::le(Spec::var(var), Spec::int(*hi)));
            }
        }

        // 2. Generate monotonicity invariants
        for var in &self.increasing_vars {
            // If we have initial value from context
            if let Some(init_val) = context.initial_values.get(var) {
                invariants.push(Spec::ge(Spec::var(var), init_val.clone()));
            }
        }

        for var in &self.decreasing_vars {
            // For decreasing variables, they stay >= 0 (common pattern)
            invariants.push(Spec::ge(Spec::var(var), Spec::int(0)));
        }

        // 3. Analyze common patterns
        if let Some(pattern_invariant) = self.detect_counter_pattern(cond, body) {
            invariants.push(pattern_invariant);
        }

        if let Some(sum_invariant) = self.detect_sum_pattern(body, context) {
            invariants.push(sum_invariant);
        }

        // 4. Include ghost variable invariants from context
        // These provide additional tracking for loop variables
        let ghost_invs = context.ghost_invariants();
        invariants.extend(ghost_invs);

        invariants
    }

    /// Analyze loop condition to extract bounds
    fn analyze_condition(&mut self, cond: &CExpr) {
        if let CExpr::BinOp { op, left, right } = cond {
            match op {
                BinOp::Lt => {
                    // i < n: upper bound on i
                    if let CExpr::Var(var) = left.as_ref() {
                        if let CExpr::Var(_bound_var) = right.as_ref() {
                            self.bounds.push((var.clone(), None, None));
                            // We don't know the numeric value, but we know i < bound_var
                        } else if let CExpr::IntLit(n) = right.as_ref() {
                            self.bounds.push((var.clone(), None, Some(*n - 1)));
                        }
                    }
                }
                BinOp::Le => {
                    // i <= n: upper bound on i (inclusive)
                    if let CExpr::Var(var) = left.as_ref() {
                        if let CExpr::IntLit(n) = right.as_ref() {
                            self.bounds.push((var.clone(), None, Some(*n)));
                        }
                    }
                }
                BinOp::Gt => {
                    // i > n: lower bound on i
                    if let CExpr::Var(var) = left.as_ref() {
                        if let CExpr::IntLit(n) = right.as_ref() {
                            self.bounds.push((var.clone(), Some(*n + 1), None));
                        }
                    }
                }
                BinOp::Ge => {
                    // i >= n: lower bound on i
                    if let CExpr::Var(var) = left.as_ref() {
                        if let CExpr::IntLit(n) = right.as_ref() {
                            self.bounds.push((var.clone(), Some(*n), None));
                        }
                    }
                }
                BinOp::LogAnd => {
                    // Conjunction: analyze both parts
                    self.analyze_condition(left);
                    self.analyze_condition(right);
                }
                _ => {}
            }
        }
    }

    /// Analyze loop body for variable modifications
    fn analyze_body(&mut self, body: &CStmt) {
        match body {
            CStmt::Block(stmts) => {
                for stmt in stmts {
                    self.analyze_body(stmt);
                }
            }
            CStmt::Expr(expr) => {
                self.analyze_modification(expr);
            }
            CStmt::If {
                then_stmt,
                else_stmt,
                ..
            } => {
                self.analyze_body(then_stmt);
                if let Some(else_s) = else_stmt {
                    self.analyze_body(else_s);
                }
            }
            CStmt::While {
                body: inner_body, ..
            }
            | CStmt::DoWhile {
                body: inner_body, ..
            } => {
                self.analyze_body(inner_body);
            }
            CStmt::For {
                body: inner_body,
                update,
                ..
            } => {
                self.analyze_body(inner_body);
                if let Some(upd) = update {
                    self.analyze_modification(upd);
                }
            }
            _ => {}
        }
    }

    /// Analyze an expression for variable modifications
    fn analyze_modification(&mut self, expr: &CExpr) {
        match expr {
            CExpr::UnaryOp { op, operand } => {
                if let CExpr::Var(var) = operand.as_ref() {
                    match op {
                        UnaryOp::PreInc | UnaryOp::PostInc => {
                            self.increasing_vars.push(var.clone());
                        }
                        UnaryOp::PreDec | UnaryOp::PostDec => {
                            self.decreasing_vars.push(var.clone());
                        }
                        _ => {}
                    }
                }
            }
            CExpr::BinOp { op, left, right } => {
                match op {
                    BinOp::AddAssign => {
                        if let CExpr::Var(var) = left.as_ref() {
                            // Check if adding a positive value
                            if let CExpr::IntLit(n) = right.as_ref() {
                                if *n > 0 {
                                    self.increasing_vars.push(var.clone());
                                } else if *n < 0 {
                                    self.decreasing_vars.push(var.clone());
                                }
                            }
                        }
                    }
                    BinOp::SubAssign => {
                        if let CExpr::Var(var) = left.as_ref() {
                            if let CExpr::IntLit(n) = right.as_ref() {
                                if *n > 0 {
                                    self.decreasing_vars.push(var.clone());
                                } else if *n < 0 {
                                    self.increasing_vars.push(var.clone());
                                }
                            }
                        }
                    }
                    BinOp::Assign => {
                        // Check for i = i + 1 pattern
                        if let CExpr::Var(var) = left.as_ref() {
                            if let CExpr::BinOp {
                                op: inner_op,
                                left: inner_left,
                                right: inner_right,
                            } = right.as_ref()
                            {
                                if let CExpr::Var(inner_var) = inner_left.as_ref() {
                                    if inner_var == var {
                                        match inner_op {
                                            BinOp::Add => {
                                                if let CExpr::IntLit(n) = inner_right.as_ref() {
                                                    if *n > 0 {
                                                        self.increasing_vars.push(var.clone());
                                                    }
                                                }
                                            }
                                            BinOp::Sub => {
                                                if let CExpr::IntLit(n) = inner_right.as_ref() {
                                                    if *n > 0 {
                                                        self.decreasing_vars.push(var.clone());
                                                    }
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Detect common counter loop pattern: for (i = 0; i < n; i++)
    fn detect_counter_pattern(&self, cond: &CExpr, _body: &CStmt) -> Option<Spec> {
        // Check if we have a variable that's both bounded and increasing
        if let CExpr::BinOp {
            op: BinOp::Lt,
            left,
            right,
        } = cond
        {
            if let CExpr::Var(var) = left.as_ref() {
                if self.increasing_vars.contains(var) {
                    // We have i < bound with i++
                    // Invariant: 0 <= i <= bound
                    let bound = match right.as_ref() {
                        CExpr::Var(bound_var) => Spec::var(bound_var),
                        CExpr::IntLit(n) => Spec::int(*n),
                        _ => return None,
                    };
                    return Some(Spec::and(vec![
                        Spec::ge(Spec::var(var), Spec::int(0)),
                        Spec::le(Spec::var(var), bound),
                    ]));
                }
            }
        }
        None
    }

    /// Detect sum accumulator pattern: `sum += arr[i]` or `sum = sum + arr[i]`
    ///
    /// Infers invariant: sum >= initial_sum (for non-negative array elements)
    pub(crate) fn detect_sum_pattern(
        &self,
        body: &CStmt,
        context: &InferenceContext,
    ) -> Option<Spec> {
        let accum_info = self.find_accumulator_pattern(body)?;

        match accum_info.op {
            AccumulatorOp::Add => {
                // sum is monotonically increasing (assuming non-negative adds)
                // Invariant: sum >= initial_sum
                if let Some(init_val) = context.initial_values.get(&accum_info.accumulator) {
                    Some(Spec::ge(
                        Spec::var(&accum_info.accumulator),
                        init_val.clone(),
                    ))
                } else {
                    // Common pattern: sum initialized to 0
                    Some(Spec::ge(Spec::var(&accum_info.accumulator), Spec::int(0)))
                }
            }
            AccumulatorOp::Mul => {
                // product can be 0 or maintain sign
                if let Some(init_val) = context.initial_values.get(&accum_info.accumulator) {
                    // product preserves non-negativity if init and multipliers are >= 0
                    Some(Spec::and(vec![
                        Spec::ge(Spec::var(&accum_info.accumulator), Spec::int(0)),
                        Spec::implies(
                            Spec::ge(init_val.clone(), Spec::int(0)),
                            Spec::ge(Spec::var(&accum_info.accumulator), Spec::int(0)),
                        ),
                    ]))
                } else {
                    None
                }
            }
            AccumulatorOp::BitwiseOr | AccumulatorOp::BitwiseAnd => {
                // Bitwise operations preserve boundedness
                Some(Spec::ge(Spec::var(&accum_info.accumulator), Spec::int(0)))
            }
            AccumulatorOp::Min => {
                // min is monotonically decreasing or equal
                if let Some(init_val) = context.initial_values.get(&accum_info.accumulator) {
                    Some(Spec::le(
                        Spec::var(&accum_info.accumulator),
                        init_val.clone(),
                    ))
                } else {
                    None
                }
            }
            AccumulatorOp::Max => {
                // max is monotonically increasing or equal
                if let Some(init_val) = context.initial_values.get(&accum_info.accumulator) {
                    Some(Spec::ge(
                        Spec::var(&accum_info.accumulator),
                        init_val.clone(),
                    ))
                } else {
                    None
                }
            }
            AccumulatorOp::Count => {
                // count is monotonically increasing and non-negative
                // count++ only increments, never decrements
                // Invariant: count >= 0 (always, since it started >= 0 and only increments)
                // And if we have loop bound n: count <= i <= n
                if let Some(init_val) = context.initial_values.get(&accum_info.accumulator) {
                    Some(Spec::ge(
                        Spec::var(&accum_info.accumulator),
                        init_val.clone(),
                    ))
                } else {
                    // Even without initial value, count is non-negative
                    Some(Spec::ge(Spec::var(&accum_info.accumulator), Spec::int(0)))
                }
            }
        }
    }

    /// Find accumulator patterns in a loop body
    fn find_accumulator_pattern(&self, body: &CStmt) -> Option<AccumulatorInfo> {
        match body {
            CStmt::Block(stmts) => {
                for stmt in stmts {
                    if let Some(info) = self.find_accumulator_pattern(stmt) {
                        return Some(info);
                    }
                }
                None
            }
            CStmt::Expr(expr) => self.analyze_accumulator_expr(expr),
            CStmt::If {
                cond,
                then_stmt,
                else_stmt,
            } => {
                // Check for min/max pattern: if (value < var) var = value;
                if let Some(info) = self.detect_minmax_pattern(cond, then_stmt) {
                    return Some(info);
                }
                // Check for counting pattern: if (condition) { count++; }
                if let Some(info) = self.detect_counting_pattern(cond, then_stmt) {
                    return Some(info);
                }
                // Look in both branches for other patterns
                if let Some(info) = self.find_accumulator_pattern(then_stmt) {
                    return Some(info);
                }
                if let Some(else_s) = else_stmt {
                    return self.find_accumulator_pattern(else_s);
                }
                None
            }
            _ => None,
        }
    }

    /// Detect min/max accumulator patterns
    ///
    /// Min pattern: `if (value < min) min = value;` or `if (min > value) min = value;`
    /// Max pattern: `if (value > max) max = value;` or `if (max < value) max = value;`
    fn detect_minmax_pattern(&self, cond: &CExpr, then_body: &CStmt) -> Option<AccumulatorInfo> {
        // Extract the comparison from condition
        let (comparison_op, left, right) = match cond {
            CExpr::BinOp { op, left, right } => (op, left.as_ref(), right.as_ref()),
            _ => return None,
        };

        // Extract the assignment from then_body
        let (assign_var, assign_value) = match then_body {
            CStmt::Expr(CExpr::BinOp {
                op: BinOp::Assign,
                left,
                right,
            }) => {
                if let CExpr::Var(var) = left.as_ref() {
                    (var, right.as_ref())
                } else {
                    return None;
                }
            }
            CStmt::Block(stmts) if stmts.len() == 1 => {
                if let CStmt::Expr(CExpr::BinOp {
                    op: BinOp::Assign,
                    left,
                    right,
                }) = &stmts[0]
                {
                    if let CExpr::Var(var) = left.as_ref() {
                        (var, right.as_ref())
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Check for min pattern: if (value < var) var = value;
        // - Condition: value < var  (where var is the accumulator)
        // - Assignment: var = value
        match comparison_op {
            BinOp::Lt => {
                // value < var => min pattern
                if let CExpr::Var(cond_var) = right {
                    if cond_var == assign_var && self.exprs_equal(left, assign_value) {
                        return Some(AccumulatorInfo {
                            accumulator: assign_var.clone(),
                            op: AccumulatorOp::Min,
                        });
                    }
                }
                // var > value => min pattern (var > value is same as value < var)
                if let CExpr::Var(cond_var) = left {
                    if cond_var == assign_var && self.exprs_equal(right, assign_value) {
                        // This is actually max pattern for `if (var > value) var = value`
                        // Wait, let me reconsider...
                        // if (var > value) var = value means var becomes smaller -> min
                        return None; // This pattern doesn't make sense for min/max
                    }
                }
            }
            BinOp::Gt => {
                // value > var => max pattern
                if let CExpr::Var(cond_var) = right {
                    if cond_var == assign_var && self.exprs_equal(left, assign_value) {
                        return Some(AccumulatorInfo {
                            accumulator: assign_var.clone(),
                            op: AccumulatorOp::Max,
                        });
                    }
                }
            }
            BinOp::Le => {
                // value <= var => min pattern (includes equal case)
                if let CExpr::Var(cond_var) = right {
                    if cond_var == assign_var && self.exprs_equal(left, assign_value) {
                        return Some(AccumulatorInfo {
                            accumulator: assign_var.clone(),
                            op: AccumulatorOp::Min,
                        });
                    }
                }
            }
            BinOp::Ge => {
                // value >= var => max pattern (includes equal case)
                if let CExpr::Var(cond_var) = right {
                    if cond_var == assign_var && self.exprs_equal(left, assign_value) {
                        return Some(AccumulatorInfo {
                            accumulator: assign_var.clone(),
                            op: AccumulatorOp::Max,
                        });
                    }
                }
            }
            _ => {}
        }

        None
    }

    /// Detect conditional counting pattern: `if (condition) { count++; }`
    ///
    /// This pattern counts elements matching a condition:
    /// ```c
    /// int count = 0;
    /// for (i = 0; i < n; i++) {
    ///     if (arr[i] > threshold) {
    ///         count++;
    ///     }
    /// }
    /// ```
    ///
    /// The key feature is that the increment is conditional, not unconditional.
    pub(crate) fn detect_counting_pattern(
        &self,
        _cond: &CExpr,
        then_body: &CStmt,
    ) -> Option<AccumulatorInfo> {
        // Look for count++ or ++count or count += 1 in the then branch
        let increment_var = self.find_simple_increment(then_body)?;

        // This is a counting pattern - the condition determines when to count
        Some(AccumulatorInfo {
            accumulator: increment_var,
            op: AccumulatorOp::Count,
        })
    }

    /// Find a simple increment pattern (var++ or ++var or var += 1) in a statement
    fn find_simple_increment(&self, stmt: &CStmt) -> Option<String> {
        match stmt {
            CStmt::Expr(expr) => self.is_increment_expr(expr),
            CStmt::Block(stmts) if stmts.len() == 1 => self.find_simple_increment(&stmts[0]),
            CStmt::Block(stmts) => {
                // Check if block contains exactly one increment
                let mut increment_var = None;
                for s in stmts {
                    if let Some(var) = self.find_simple_increment(s) {
                        if increment_var.is_some() {
                            return None; // Multiple increments, not a simple count
                        }
                        increment_var = Some(var);
                    }
                }
                increment_var
            }
            _ => None,
        }
    }

    /// Check if an expression is a simple increment (var++, ++var, var += 1)
    pub(crate) fn is_increment_expr(&self, expr: &CExpr) -> Option<String> {
        match expr {
            // var++ or ++var
            CExpr::UnaryOp {
                op: UnaryOp::PostInc | UnaryOp::PreInc,
                operand,
            } => {
                if let CExpr::Var(var) = operand.as_ref() {
                    Some(var.clone())
                } else {
                    None
                }
            }
            // var += 1
            CExpr::BinOp {
                op: BinOp::AddAssign,
                left,
                right,
            } => {
                if let (CExpr::Var(var), CExpr::IntLit(1)) = (left.as_ref(), right.as_ref()) {
                    Some(var.clone())
                } else {
                    None
                }
            }
            // var = var + 1
            CExpr::BinOp {
                op: BinOp::Assign,
                left,
                right,
            } => {
                if let CExpr::Var(var) = left.as_ref() {
                    if let CExpr::BinOp {
                        op: BinOp::Add,
                        left: add_l,
                        right: add_r,
                    } = right.as_ref()
                    {
                        // var = var + 1
                        if let CExpr::Var(v) = add_l.as_ref() {
                            if v == var {
                                if let CExpr::IntLit(1) = add_r.as_ref() {
                                    return Some(var.clone());
                                }
                            }
                        }
                        // var = 1 + var
                        if let CExpr::IntLit(1) = add_l.as_ref() {
                            if let CExpr::Var(v) = add_r.as_ref() {
                                if v == var {
                                    return Some(var.clone());
                                }
                            }
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Check if two C expressions are structurally equal
    pub(crate) fn exprs_equal(&self, a: &CExpr, b: &CExpr) -> bool {
        match (a, b) {
            (CExpr::Var(va), CExpr::Var(vb)) => va == vb,
            (CExpr::IntLit(na), CExpr::IntLit(nb)) => na == nb,
            (
                CExpr::Index {
                    array: aa,
                    index: ia,
                },
                CExpr::Index {
                    array: ab,
                    index: ib,
                },
            ) => self.exprs_equal(aa, ab) && self.exprs_equal(ia, ib),
            (
                CExpr::Member {
                    object: oa,
                    field: fa,
                },
                CExpr::Member {
                    object: ob,
                    field: fb,
                },
            ) => self.exprs_equal(oa, ob) && fa == fb,
            (
                CExpr::BinOp {
                    op: opa,
                    left: la,
                    right: ra,
                },
                CExpr::BinOp {
                    op: opb,
                    left: lb,
                    right: rb,
                },
            ) => opa == opb && self.exprs_equal(la, lb) && self.exprs_equal(ra, rb),
            (
                CExpr::UnaryOp {
                    op: opa,
                    operand: oa,
                },
                CExpr::UnaryOp {
                    op: opb,
                    operand: ob,
                },
            ) => opa == opb && self.exprs_equal(oa, ob),
            _ => false,
        }
    }

    /// Analyze an expression for accumulator patterns
    fn analyze_accumulator_expr(&self, expr: &CExpr) -> Option<AccumulatorInfo> {
        match expr {
            // sum += value
            CExpr::BinOp {
                op: BinOp::AddAssign,
                left,
                ..
            } => {
                if let CExpr::Var(var) = left.as_ref() {
                    return Some(AccumulatorInfo {
                        accumulator: var.clone(),
                        op: AccumulatorOp::Add,
                    });
                }
                None
            }
            // product *= value
            CExpr::BinOp {
                op: BinOp::MulAssign,
                left,
                ..
            } => {
                if let CExpr::Var(var) = left.as_ref() {
                    return Some(AccumulatorInfo {
                        accumulator: var.clone(),
                        op: AccumulatorOp::Mul,
                    });
                }
                None
            }
            // flags |= value
            CExpr::BinOp {
                op: BinOp::BitOrAssign,
                left,
                ..
            } => {
                if let CExpr::Var(var) = left.as_ref() {
                    return Some(AccumulatorInfo {
                        accumulator: var.clone(),
                        op: AccumulatorOp::BitwiseOr,
                    });
                }
                None
            }
            // mask &= value
            CExpr::BinOp {
                op: BinOp::BitAndAssign,
                left,
                ..
            } => {
                if let CExpr::Var(var) = left.as_ref() {
                    return Some(AccumulatorInfo {
                        accumulator: var.clone(),
                        op: AccumulatorOp::BitwiseAnd,
                    });
                }
                None
            }
            // sum = sum + value
            CExpr::BinOp {
                op: BinOp::Assign,
                left,
                right,
            } => {
                if let CExpr::Var(var) = left.as_ref() {
                    if let CExpr::BinOp {
                        op: inner_op,
                        left: inner_left,
                        ..
                    } = right.as_ref()
                    {
                        if let CExpr::Var(inner_var) = inner_left.as_ref() {
                            if inner_var == var {
                                return match inner_op {
                                    BinOp::Add => Some(AccumulatorInfo {
                                        accumulator: var.clone(),
                                        op: AccumulatorOp::Add,
                                    }),
                                    BinOp::Mul => Some(AccumulatorInfo {
                                        accumulator: var.clone(),
                                        op: AccumulatorOp::Mul,
                                    }),
                                    _ => None,
                                };
                            }
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Detect search loop pattern: while (!found && i < n)
    ///
    /// Infers invariants about partial search results
    pub fn detect_search_pattern(&self, cond: &CExpr, body: &CStmt) -> Option<Spec> {
        // Look for: !found && i < n  or  found == 0 && i < n
        let search_info = self.analyze_search_condition(cond)?;

        // Look for found = 1 or found = true in body
        if self.has_found_assignment(body, &search_info.found_var) {
            // Invariant:
            // 1. If not found, we've checked all elements before i
            // 2. The index is within bounds
            let idx_bound = if let Some(bound) = &search_info.bound {
                Spec::and(vec![
                    Spec::ge(Spec::var(&search_info.index_var), Spec::int(0)),
                    Spec::le(Spec::var(&search_info.index_var), bound.clone()),
                ])
            } else {
                Spec::ge(Spec::var(&search_info.index_var), Spec::int(0))
            };

            // found_var is either 0 or 1 (boolean-like)
            let found_bounded = Spec::and(vec![
                Spec::ge(Spec::var(&search_info.found_var), Spec::int(0)),
                Spec::le(Spec::var(&search_info.found_var), Spec::int(1)),
            ]);

            Some(Spec::and(vec![idx_bound, found_bounded]))
        } else {
            None
        }
    }

    /// Analyze a condition for search loop pattern
    fn analyze_search_condition(&self, cond: &CExpr) -> Option<SearchInfo> {
        match cond {
            // !found && i < n
            CExpr::BinOp {
                op: BinOp::LogAnd,
                left,
                right,
            } => {
                // Check left for !found
                let found_var = self.get_negated_var(left)?;

                // Check right for i < n
                if let CExpr::BinOp {
                    op: BinOp::Lt,
                    left: idx,
                    right: bound,
                } = right.as_ref()
                {
                    if let CExpr::Var(index_var) = idx.as_ref() {
                        let bound_spec = self.expr_to_simple_spec(bound);
                        return Some(SearchInfo {
                            found_var,
                            index_var: index_var.clone(),
                            bound: bound_spec,
                        });
                    }
                }
                None
            }
            // found == 0 && i < n  (alternative pattern)
            _ => None,
        }
    }

    /// Get variable name from !var or var == 0
    fn get_negated_var(&self, expr: &CExpr) -> Option<String> {
        match expr {
            CExpr::UnaryOp {
                op: UnaryOp::LogNot,
                operand,
            } => {
                if let CExpr::Var(var) = operand.as_ref() {
                    return Some(var.clone());
                }
                None
            }
            CExpr::BinOp {
                op: BinOp::Eq,
                left,
                right,
            } => {
                if let CExpr::Var(var) = left.as_ref() {
                    if matches!(right.as_ref(), CExpr::IntLit(0)) {
                        return Some(var.clone());
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Check if body contains found = 1 or found = true
    fn has_found_assignment(&self, body: &CStmt, found_var: &str) -> bool {
        match body {
            CStmt::Block(stmts) => stmts
                .iter()
                .any(|s| self.has_found_assignment(s, found_var)),
            CStmt::Expr(expr) => self.is_found_assignment(expr, found_var),
            CStmt::If {
                then_stmt,
                else_stmt,
                ..
            } => {
                self.has_found_assignment(then_stmt, found_var)
                    || else_stmt
                        .as_ref()
                        .is_some_and(|s| self.has_found_assignment(s, found_var))
            }
            _ => false,
        }
    }

    /// Check if expr is found = 1 or similar
    fn is_found_assignment(&self, expr: &CExpr, found_var: &str) -> bool {
        if let CExpr::BinOp {
            op: BinOp::Assign,
            left,
            right,
        } = expr
        {
            if let CExpr::Var(var) = left.as_ref() {
                if var == found_var {
                    // Check for = 1 or = true
                    return matches!(right.as_ref(), CExpr::IntLit(1 | _));
                }
            }
        }
        false
    }

    /// Convert simple C expression to Spec (for bounds)
    fn expr_to_simple_spec(&self, expr: &CExpr) -> Option<Spec> {
        match expr {
            CExpr::Var(name) => Some(Spec::var(name)),
            CExpr::IntLit(n) => Some(Spec::int(*n)),
            _ => None,
        }
    }

    /// Clear state for analyzing a new loop
    pub fn reset(&mut self) {
        self.increasing_vars.clear();
        self.decreasing_vars.clear();
        self.bounds.clear();
    }
}

/// Information about an accumulator pattern
pub(crate) struct AccumulatorInfo {
    /// The accumulator variable name
    pub(crate) accumulator: String,
    /// The accumulation operation
    pub(crate) op: AccumulatorOp,
}

/// Types of accumulator operations
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum AccumulatorOp {
    Add,
    Mul,
    BitwiseOr,
    BitwiseAnd,
    Min,
    Max,
    /// Conditional counting: count++ inside if (condition)
    Count,
}

/// Information about a search loop pattern
struct SearchInfo {
    /// The "found" flag variable
    found_var: String,
    /// The index variable
    index_var: String,
    /// The upper bound (if known)
    bound: Option<Spec>,
}

/// Context for loop invariant inference
#[derive(Default)]
pub struct InferenceContext {
    /// Initial values of variables before the loop
    pub initial_values: HashMap<String, Spec>,
    /// Function preconditions that may be useful
    pub preconditions: Vec<Spec>,
    /// Postcondition that the loop must establish
    pub postcondition: Option<Spec>,
    /// Ghost variables for tracking auxiliary properties
    pub ghost_vars: Vec<GhostVariable>,
}

impl InferenceContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an initial value for a variable
    pub fn set_initial(&mut self, var: &str, value: Spec) {
        self.initial_values.insert(var.to_string(), value);
    }

    /// Add a ghost variable for tracking a property
    pub fn add_ghost(&mut self, ghost: GhostVariable) {
        self.ghost_vars.push(ghost);
    }

    /// Generate ghost variable declarations for a loop
    ///
    /// Common ghost variables:
    /// - Loop iteration counter (implicit in for loops)
    /// - Original values of modified variables
    /// - Partial results (partial sum, partial product)
    pub fn generate_loop_ghosts(&mut self, loop_var: Option<&str>, modified_vars: &[String]) {
        // Ghost for loop iteration count
        if let Some(var) = loop_var {
            self.ghost_vars.push(GhostVariable {
                name: format!("__ghost_iter_{var}"),
                kind: GhostKind::IterationCount,
                related_var: Some(var.to_string()),
                initial_value: Some(Spec::int(0)),
                array_name: None,
                array_index: None,
            });
        }

        // Ghost for original values of modified variables
        for var in modified_vars {
            let ghost_name = format!("__ghost_old_{var}");
            self.ghost_vars.push(GhostVariable {
                name: ghost_name.clone(),
                kind: GhostKind::OriginalValue,
                related_var: Some(var.clone()),
                initial_value: self.initial_values.get(var).cloned(),
                array_name: None,
                array_index: None,
            });
        }
    }

    /// Add a ghost for tracking an array element
    ///
    /// This is useful for proving frame conditions - that unmodified
    /// array elements retain their original values.
    pub fn add_array_ghost(&mut self, arr_name: &str, index: Spec, init: Option<Spec>) {
        self.ghost_vars
            .push(GhostVariable::array_element(arr_name, index, init));
    }

    /// Generate ghosts for array elements that are read but not written
    ///
    /// This helps prove frame conditions for array loops - elements not
    /// touched by the loop retain their initial values.
    pub fn generate_array_frame_ghosts(&mut self, accesses: &[(String, Spec, bool)]) {
        use std::collections::HashSet;

        // Collect arrays that are written to (we can't add frame ghosts for these)
        let mut written_patterns: HashSet<(String, String)> = HashSet::new();
        for (arr, idx, is_write) in accesses {
            if *is_write {
                // Create a pattern key for the written access
                let idx_key = Self::index_to_key(idx);
                written_patterns.insert((arr.clone(), idx_key));
            }
        }

        // For read accesses with constant indices, add frame ghosts if not written
        for (arr, idx, is_write) in accesses {
            if !*is_write {
                let idx_key = Self::index_to_key(idx);
                if !written_patterns.contains(&(arr.clone(), idx_key.clone())) {
                    // This element is only read, add a frame ghost
                    self.ghost_vars.push(GhostVariable::array_element(
                        arr,
                        idx.clone(),
                        None, // Initial value would need to be looked up from context
                    ));
                }
            }
        }
    }

    /// Helper to create a string key for an index expression
    fn index_to_key(spec: &Spec) -> String {
        match spec {
            Spec::Var(v) => v.clone(),
            Spec::Int(n) => n.to_string(),
            _ => format!("{spec:?}"), // Fallback for complex expressions
        }
    }

    /// Get ghost invariants based on tracked ghost variables
    pub fn ghost_invariants(&self) -> Vec<Spec> {
        let mut invs = Vec::new();

        for ghost in &self.ghost_vars {
            match ghost.kind {
                GhostKind::IterationCount => {
                    // Iteration count is non-negative
                    invs.push(Spec::ge(Spec::var(&ghost.name), Spec::int(0)));
                }
                GhostKind::OriginalValue => {
                    // Original value ghost is constant (equals initial value)
                    if let (Some(ref _related), Some(ref init)) =
                        (&ghost.related_var, &ghost.initial_value)
                    {
                        // __ghost_old_x == original_x_value
                        invs.push(Spec::eq(Spec::var(&ghost.name), init.clone()));
                        // Track that current value relates to original
                        // This is useful for proving properties like "x >= __ghost_old_x"
                    }
                }
                GhostKind::PartialResult => {
                    // Partial results grow monotonically (for sum/product)
                    if let Some(ref init) = ghost.initial_value {
                        invs.push(Spec::ge(Spec::var(&ghost.name), init.clone()));
                    }
                }
                GhostKind::ArrayElement => {
                    // Array element ghost tracks initial value of arr[idx]
                    // The invariant is: __ghost_arr_name_idx == original_arr[idx]
                    if let (Some(ref arr), Some(ref idx), Some(ref init)) =
                        (&ghost.array_name, &ghost.array_index, &ghost.initial_value)
                    {
                        // Ghost equals its initial value (frame condition)
                        invs.push(Spec::eq(Spec::var(&ghost.name), init.clone()));
                        // Also useful: current arr[idx] relates to ghost
                        // This can help prove arr[idx] unchanged for unmodified indices
                        let arr_access = Spec::Index {
                            base: Box::new(Spec::var(arr)),
                            index: Box::new(idx.clone()),
                        };
                        // If index is unchanged, arr[idx] == __ghost_arr_name_idx
                        invs.push(Spec::eq(arr_access, Spec::var(&ghost.name)));
                    }
                }
                GhostKind::Custom => {
                    // Custom ghosts are handled by user-provided invariants
                }
            }
        }

        invs
    }
}

/// A ghost variable for tracking auxiliary properties in loop invariants
#[derive(Debug, Clone)]
pub struct GhostVariable {
    /// Name of the ghost variable (prefixed with __ghost_)
    pub name: String,
    /// Kind of ghost variable
    pub kind: GhostKind,
    /// Related program variable (if any)
    pub related_var: Option<String>,
    /// Initial value of the ghost variable
    pub initial_value: Option<Spec>,
    /// For ArrayElement: the array name
    pub array_name: Option<String>,
    /// For ArrayElement: the index expression (as spec)
    pub array_index: Option<Spec>,
}

impl GhostVariable {
    /// Create a new iteration count ghost
    pub fn iteration_count(loop_var: &str) -> Self {
        GhostVariable {
            name: format!("__ghost_iter_{loop_var}"),
            kind: GhostKind::IterationCount,
            related_var: Some(loop_var.to_string()),
            initial_value: Some(Spec::int(0)),
            array_name: None,
            array_index: None,
        }
    }

    /// Create a ghost for the original value of a variable
    pub fn original_value(var: &str, init: Option<Spec>) -> Self {
        GhostVariable {
            name: format!("__ghost_old_{var}"),
            kind: GhostKind::OriginalValue,
            related_var: Some(var.to_string()),
            initial_value: init,
            array_name: None,
            array_index: None,
        }
    }

    /// Create a ghost for a partial result (e.g., partial sum)
    pub fn partial_result(name: &str, init: Spec) -> Self {
        GhostVariable {
            name: format!("__ghost_partial_{name}"),
            kind: GhostKind::PartialResult,
            related_var: Some(name.to_string()),
            initial_value: Some(init),
            array_name: None,
            array_index: None,
        }
    }

    /// Create a ghost for tracking an array element value
    ///
    /// # Arguments
    /// * `arr_name` - Name of the array variable
    /// * `index` - Index expression (as Spec)
    /// * `init` - Initial value at that index (if known)
    pub fn array_element(arr_name: &str, index: Spec, init: Option<Spec>) -> Self {
        // Create a unique name based on array and a simple index representation
        let index_repr = match &index {
            Spec::Var(v) => v.clone(),
            Spec::Int(n) => n.to_string(),
            _ => "expr".to_string(),
        };
        GhostVariable {
            name: format!("__ghost_arr_{arr_name}_{index_repr}"),
            kind: GhostKind::ArrayElement,
            related_var: None,
            initial_value: init,
            array_name: Some(arr_name.to_string()),
            array_index: Some(index),
        }
    }

    /// Create a custom ghost variable
    pub fn custom(name: &str) -> Self {
        GhostVariable {
            name: name.to_string(),
            kind: GhostKind::Custom,
            related_var: None,
            initial_value: None,
            array_name: None,
            array_index: None,
        }
    }
}

/// Kind of ghost variable
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GhostKind {
    /// Tracks the number of loop iterations
    IterationCount,
    /// Tracks the original value of a variable
    OriginalValue,
    /// Tracks a partial result (sum, product, etc.)
    PartialResult,
    /// Tracks an array element at a specific index
    ArrayElement,
    /// User-defined ghost variable
    Custom,
}
