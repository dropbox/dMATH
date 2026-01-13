//! Collection and extraction methods for verification condition generation
//!
//! This module contains methods for collecting modified locations, local variables,
//! switch cases, loop variables, array accesses, etc.

use crate::expr::{BinOp, CExpr, UnaryOp};
use crate::spec::{Location, Spec};
use crate::stmt::{CStmt, CaseLabel};

use super::{ModifiedLocation, VCGen};

impl VCGen {
    /// Collect all locations modified by a statement
    pub fn collect_modified_locations(&self, stmt: &CStmt) -> Vec<ModifiedLocation> {
        let mut locations = Vec::new();
        self.collect_modified_recursive(stmt, &mut locations);
        locations
    }

    fn collect_modified_recursive(&self, stmt: &CStmt, locations: &mut Vec<ModifiedLocation>) {
        match stmt {
            CStmt::Empty
            | CStmt::Break
            | CStmt::Continue
            | CStmt::Goto(_)
            | CStmt::Return(None)
            | CStmt::Asm(_)
            | CStmt::Assert(_)
            | CStmt::Assume(_) => {}

            CStmt::Expr(e) | CStmt::Return(Some(e)) => {
                self.collect_expr_modifications(e, locations);
            }

            CStmt::Decl(decl) => {
                // Local variable declaration - modifies the local
                locations.push(ModifiedLocation {
                    location: Location::Deref(Spec::var(&decl.name)),
                    description: format!("local variable '{}'", decl.name),
                    source_line: None,
                });
            }

            CStmt::DeclList(decls) => {
                for decl in decls {
                    locations.push(ModifiedLocation {
                        location: Location::Deref(Spec::var(&decl.name)),
                        description: format!("local variable '{}'", decl.name),
                        source_line: None,
                    });
                }
            }

            CStmt::Block(stmts) => {
                for s in stmts {
                    self.collect_modified_recursive(s, locations);
                }
            }

            CStmt::If {
                cond,
                then_stmt,
                else_stmt,
            } => {
                self.collect_expr_modifications(cond, locations);
                self.collect_modified_recursive(then_stmt, locations);
                if let Some(e) = else_stmt {
                    self.collect_modified_recursive(e, locations);
                }
            }

            CStmt::Switch { cond, body }
            | CStmt::While { cond, body }
            | CStmt::DoWhile { body, cond } => {
                self.collect_expr_modifications(cond, locations);
                self.collect_modified_recursive(body, locations);
            }

            CStmt::Case { stmt, .. } | CStmt::Label { stmt, .. } => {
                self.collect_modified_recursive(stmt, locations);
            }

            CStmt::For {
                init,
                cond,
                update,
                body,
            } => {
                if let Some(init) = init {
                    self.collect_modified_recursive(init, locations);
                }
                if let Some(cond) = cond {
                    self.collect_expr_modifications(cond, locations);
                }
                if let Some(update) = update {
                    self.collect_expr_modifications(update, locations);
                }
                self.collect_modified_recursive(body, locations);
            }

            CStmt::FuncDef(func) => {
                self.collect_modified_recursive(&func.body, locations);
            }
        }
    }

    fn collect_expr_modifications(&self, expr: &CExpr, locations: &mut Vec<ModifiedLocation>) {
        match expr {
            CExpr::BinOp { op, left, right } if op.is_assignment() => {
                // Assignment - collect the LHS as modified location
                if let Some(loc) = self.expr_to_location(left) {
                    locations.push(ModifiedLocation {
                        location: loc,
                        description: format!("assignment to '{}'", self.expr_description(left)),
                        source_line: None,
                    });
                }
                // Also check right side for nested modifications
                self.collect_expr_modifications(right, locations);
            }

            CExpr::UnaryOp { op, operand } if op.is_inc_dec() => {
                // Increment/decrement - modifies the operand
                if let Some(loc) = self.expr_to_location(operand) {
                    locations.push(ModifiedLocation {
                        location: loc,
                        description: format!(
                            "increment/decrement of '{}'",
                            self.expr_description(operand)
                        ),
                        source_line: None,
                    });
                }
            }

            // Recurse into subexpressions
            CExpr::BinOp { left, right, .. } => {
                self.collect_expr_modifications(left, locations);
                self.collect_expr_modifications(right, locations);
            }

            CExpr::UnaryOp { operand, .. } => {
                self.collect_expr_modifications(operand, locations);
            }

            CExpr::Conditional {
                cond,
                then_expr,
                else_expr,
            } => {
                self.collect_expr_modifications(cond, locations);
                self.collect_expr_modifications(then_expr, locations);
                self.collect_expr_modifications(else_expr, locations);
            }

            CExpr::Cast { expr, .. } => {
                self.collect_expr_modifications(expr, locations);
            }

            CExpr::Call { func, args } => {
                self.collect_expr_modifications(func, locations);
                for arg in args {
                    self.collect_expr_modifications(arg, locations);
                }
                // Interprocedural analysis: include callee's assigns clause with parameter substitution
                if let CExpr::Var(func_name) = func.as_ref() {
                    if let Some(callee_spec) = self.func_specs.get(func_name).cloned() {
                        // Convert actual arguments to Spec for substitution
                        let arg_specs: Vec<Spec> =
                            args.iter().map(|a| self.expr_to_spec(a)).collect();

                        for loc in &callee_spec.assigns {
                            // Substitute formal parameters with actual arguments in assigns locations
                            let subst_loc =
                                self.subst_params_in_location(loc, &callee_spec.params, &arg_specs);
                            locations.push(ModifiedLocation {
                                location: subst_loc,
                                description: format!("callee '{func_name}' assigns clause"),
                                source_line: None,
                            });
                        }
                    }
                }
            }

            CExpr::Index { array, index } => {
                self.collect_expr_modifications(array, locations);
                self.collect_expr_modifications(index, locations);
            }

            CExpr::Member { object, .. } => {
                self.collect_expr_modifications(object, locations);
            }

            CExpr::Arrow { pointer, .. } => {
                self.collect_expr_modifications(pointer, locations);
            }

            CExpr::StmtExpr(stmts) => {
                for s in stmts {
                    self.collect_modified_recursive(s, locations);
                }
            }

            // Literals and simple expressions don't modify memory
            CExpr::IntLit(_)
            | CExpr::UIntLit(_)
            | CExpr::FloatLit(_)
            | CExpr::CharLit(_)
            | CExpr::StringLit(_)
            | CExpr::Var(_)
            | CExpr::SizeOf(_)
            | CExpr::AlignOf(_)
            | CExpr::CompoundLiteral { .. }
            | CExpr::Generic { .. } => {}
        }
    }

    /// Collect all local variable names declared in a statement
    pub fn collect_local_variables(&self, stmt: &CStmt) -> Vec<String> {
        let mut locals = Vec::new();
        self.collect_locals_recursive(stmt, &mut locals);
        locals
    }

    fn collect_locals_recursive(&self, stmt: &CStmt, locals: &mut Vec<String>) {
        match stmt {
            CStmt::Empty
            | CStmt::Break
            | CStmt::Continue
            | CStmt::Goto(_)
            | CStmt::Expr(_)
            | CStmt::Return(_)
            | CStmt::Asm(_)
            | CStmt::Assert(_)
            | CStmt::Assume(_) => {}

            CStmt::Decl(decl) => {
                locals.push(decl.name.clone());
            }

            CStmt::DeclList(decls) => {
                for decl in decls {
                    locals.push(decl.name.clone());
                }
            }

            CStmt::Block(stmts) => {
                for s in stmts {
                    self.collect_locals_recursive(s, locals);
                }
            }

            CStmt::If {
                then_stmt,
                else_stmt,
                ..
            } => {
                self.collect_locals_recursive(then_stmt, locals);
                if let Some(e) = else_stmt {
                    self.collect_locals_recursive(e, locals);
                }
            }

            CStmt::Switch { body, .. }
            | CStmt::While { body, .. }
            | CStmt::DoWhile { body, .. } => {
                self.collect_locals_recursive(body, locals);
            }

            CStmt::Case { stmt, .. } | CStmt::Label { stmt, .. } => {
                self.collect_locals_recursive(stmt, locals);
            }

            CStmt::For { init, body, .. } => {
                if let Some(init) = init {
                    self.collect_locals_recursive(init, locals);
                }
                self.collect_locals_recursive(body, locals);
            }

            CStmt::FuncDef(func) => {
                // Nested function - locals are scoped within
                self.collect_locals_recursive(&func.body, locals);
            }
        }
    }

    /// Filter modified locations to exclude local variables
    pub fn filter_non_locals(
        &self,
        modified: Vec<ModifiedLocation>,
        locals: &[String],
    ) -> Vec<ModifiedLocation> {
        modified
            .into_iter()
            .filter(|m| {
                // Check if this is a local variable
                if let Location::Deref(Spec::Var(name)) = &m.location {
                    !locals.contains(name)
                } else {
                    true
                }
            })
            .collect()
    }

    /// Convert a C expression (lvalue) to a Location
    fn expr_to_location(&self, expr: &CExpr) -> Option<Location> {
        match expr {
            CExpr::Var(name) => Some(Location::Deref(Spec::var(name))),
            CExpr::UnaryOp {
                op: UnaryOp::Deref,
                operand,
            } => {
                // *ptr - the location is the dereferenced pointer
                let spec = Self::cexpr_to_spec(operand)?;
                Some(Location::Deref(spec))
            }
            CExpr::Index { array, index } => {
                // arr[i] - equivalent to *(arr + i)
                let base = Self::cexpr_to_spec(array)?;
                let idx = Self::cexpr_to_spec(index)?;
                Some(Location::Deref(Spec::binop(BinOp::Add, base, idx)))
            }
            CExpr::Member { object, field } => {
                // s.field - for now, treat as deref of (object + field_offset)
                // This is simplified; proper handling needs type info
                let obj_spec = Self::cexpr_to_spec(object)?;
                Some(Location::Deref(Spec::binop(
                    BinOp::Add,
                    obj_spec,
                    Spec::var(field),
                )))
            }
            CExpr::Arrow { pointer, field } => {
                // p->field = (*p).field
                let ptr_spec = Self::cexpr_to_spec(pointer)?;
                Some(Location::Deref(Spec::binop(
                    BinOp::Add,
                    ptr_spec,
                    Spec::var(field),
                )))
            }
            _ => None,
        }
    }

    /// Generate a short description of an expression for error messages
    fn expr_description(&self, expr: &CExpr) -> String {
        match expr {
            CExpr::Var(name) => name.clone(),
            CExpr::UnaryOp {
                op: UnaryOp::Deref,
                operand,
            } => format!("*{}", self.expr_description(operand)),
            CExpr::Index { array, index } => {
                format!(
                    "{}[{}]",
                    self.expr_description(array),
                    self.expr_description(index)
                )
            }
            CExpr::Member { object, field } => {
                format!("{}.{}", self.expr_description(object), field)
            }
            CExpr::Arrow { pointer, field } => {
                format!("{}->{}", self.expr_description(pointer), field)
            }
            CExpr::IntLit(n) => n.to_string(),
            _ => "expr".to_string(),
        }
    }

    /// Extract case labels and their bodies from a switch body
    ///
    /// A switch body is typically a Block containing Case statements.
    /// This function flattens the structure and handles fallthrough
    /// by combining statements until a break is encountered.
    pub(crate) fn extract_switch_cases(&self, body: &CStmt) -> Vec<(CaseLabel, CStmt)> {
        let mut cases: Vec<(CaseLabel, CStmt)> = Vec::new();

        // Collect raw case statements
        let mut raw_cases: Vec<(CaseLabel, Vec<CStmt>)> = Vec::new();
        self.collect_cases_recursive(body, &mut raw_cases);

        // Process fallthrough: combine statements until break
        for (label, stmts) in raw_cases {
            // Collect all statements, stopping at break
            let mut body_stmts = Vec::new();
            let mut has_break = false;

            for stmt in stmts {
                if matches!(stmt, CStmt::Break) {
                    has_break = true;
                    break;
                }
                body_stmts.push(stmt);
            }

            // If no break found, this case falls through
            // For WP purposes, we treat it as just the collected statements
            // (a more sophisticated analysis would track fallthrough chains)

            let case_body = if body_stmts.is_empty() {
                CStmt::Empty
            } else if body_stmts.len() == 1 {
                body_stmts.pop().unwrap()
            } else {
                CStmt::Block(body_stmts)
            };

            cases.push((label, case_body));

            // Mark whether this case has break (for potential future use)
            let _ = has_break;
        }

        cases
    }

    /// Helper to recursively collect case labels and their statements
    fn collect_cases_recursive(&self, stmt: &CStmt, cases: &mut Vec<(CaseLabel, Vec<CStmt>)>) {
        match stmt {
            CStmt::Block(stmts) => {
                // Track current case and its statements
                let mut current_label: Option<CaseLabel> = None;
                let mut current_stmts: Vec<CStmt> = Vec::new();

                for s in stmts {
                    match s {
                        CStmt::Case { label, stmt: inner } => {
                            // Save previous case if any
                            if let Some(lbl) = current_label.take() {
                                cases.push((lbl, std::mem::take(&mut current_stmts)));
                            }

                            // Start new case
                            current_label = Some(label.clone());

                            // Recursively process the inner statement
                            // which might be another case or actual code
                            self.flatten_case_stmt(inner, &mut current_stmts);
                        }
                        _ => {
                            // Non-case statement - add to current case if any
                            if current_label.is_some() {
                                current_stmts.push(s.clone());
                            }
                        }
                    }
                }

                // Save last case
                if let Some(lbl) = current_label {
                    cases.push((lbl, current_stmts));
                }
            }
            CStmt::Case { label, stmt } => {
                // Direct case (not in block)
                let mut stmts = Vec::new();
                self.flatten_case_stmt(stmt, &mut stmts);
                cases.push((label.clone(), stmts));
            }
            _ => {
                // Not a case structure - nothing to extract
            }
        }
    }

    /// Flatten nested case statements into a list of statements
    fn flatten_case_stmt(&self, stmt: &CStmt, stmts: &mut Vec<CStmt>) {
        match stmt {
            CStmt::Case {
                label: _,
                stmt: inner,
            } => {
                // Nested case (like case 1: case 2: stmt)
                // This is handled by the caller - we just record that we saw it
                // and process the inner statement
                self.flatten_case_stmt(inner, stmts);
            }
            CStmt::Block(block_stmts) => {
                // Add all block statements
                for s in block_stmts {
                    stmts.push(s.clone());
                }
            }
            _ => {
                stmts.push(stmt.clone());
            }
        }
    }

    /// Extract the loop counter variable from a loop condition
    ///
    /// Looks for patterns like `i < n`, `i <= n`, `i != n` to find
    /// the iteration variable.
    pub(crate) fn extract_loop_variable(cond: &CExpr) -> Option<String> {
        match cond {
            CExpr::BinOp { op, left, .. } => {
                match op {
                    BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Ne => {
                        if let CExpr::Var(var) = left.as_ref() {
                            return Some(var.clone());
                        }
                    }
                    _ => {}
                }
                // Also check right side for cases like `n > i`
                if let CExpr::BinOp { right, .. } = cond {
                    if let CExpr::Var(var) = right.as_ref() {
                        return Some(var.clone());
                    }
                }
                None
            }
            CExpr::UnaryOp {
                op: UnaryOp::LogNot,
                operand,
            } => {
                // !found pattern - the operand is the flag variable
                if let CExpr::Var(var) = operand.as_ref() {
                    return Some(var.clone());
                }
                None
            }
            _ => None,
        }
    }

    /// Extract variables that are modified in a statement
    ///
    /// Returns a list of variable names that are assigned to in the statement.
    pub(crate) fn extract_modified_variables(stmt: &CStmt) -> Vec<String> {
        let mut vars = Vec::new();
        Self::collect_modified_vars(stmt, &mut vars);
        vars
    }

    /// Helper to collect modified variables recursively
    fn collect_modified_vars(stmt: &CStmt, vars: &mut Vec<String>) {
        match stmt {
            CStmt::Expr(expr) => Self::collect_modified_vars_expr(expr, vars),
            CStmt::Block(stmts) => {
                for s in stmts {
                    Self::collect_modified_vars(s, vars);
                }
            }
            CStmt::If {
                then_stmt,
                else_stmt,
                ..
            } => {
                Self::collect_modified_vars(then_stmt, vars);
                if let Some(else_s) = else_stmt {
                    Self::collect_modified_vars(else_s, vars);
                }
            }
            CStmt::While { body, .. } | CStmt::DoWhile { body, .. } => {
                Self::collect_modified_vars(body, vars);
            }
            CStmt::For {
                init, update, body, ..
            } => {
                if let Some(i) = init {
                    Self::collect_modified_vars(i, vars);
                }
                if let Some(u) = update {
                    Self::collect_modified_vars_expr(u, vars);
                }
                Self::collect_modified_vars(body, vars);
            }
            // Catch-all for statements that don't modify variables or need special handling
            CStmt::Return(_)
            | CStmt::Break
            | CStmt::Continue
            | CStmt::Empty
            | CStmt::Decl(_)
            | CStmt::DeclList(_)
            | CStmt::Goto(_)
            | CStmt::Asm { .. }
            | CStmt::FuncDef(_)
            | CStmt::Assert(_)
            | CStmt::Assume(_) => {}
            // Switch/Case/Label need body traversal
            CStmt::Switch { body, .. } => Self::collect_modified_vars(body, vars),
            CStmt::Case { stmt, .. } | CStmt::Label { stmt, .. } => {
                Self::collect_modified_vars(stmt, vars);
            }
        }
    }

    /// Helper to collect modified variables from an expression
    fn collect_modified_vars_expr(expr: &CExpr, vars: &mut Vec<String>) {
        match expr {
            CExpr::BinOp {
                op:
                    BinOp::Assign
                    | BinOp::AddAssign
                    | BinOp::SubAssign
                    | BinOp::MulAssign
                    | BinOp::DivAssign
                    | BinOp::ModAssign
                    | BinOp::BitAndAssign
                    | BinOp::BitOrAssign
                    | BinOp::BitXorAssign,
                left,
                ..
            } => {
                if let CExpr::Var(var) = left.as_ref() {
                    if !vars.contains(var) {
                        vars.push(var.clone());
                    }
                }
            }
            CExpr::UnaryOp {
                op: UnaryOp::PreInc | UnaryOp::PostInc | UnaryOp::PreDec | UnaryOp::PostDec,
                operand,
            } => {
                if let CExpr::Var(var) = operand.as_ref() {
                    if !vars.contains(var) {
                        vars.push(var.clone());
                    }
                }
            }
            _ => {}
        }
    }

    /// Extract array accesses from a statement
    /// Returns list of (array_name, index_expr, is_write)
    pub(crate) fn extract_array_accesses(stmt: &CStmt) -> Vec<(String, CExpr, bool)> {
        let mut accesses = Vec::new();
        Self::collect_array_accesses(stmt, &mut accesses);
        accesses
    }

    /// Convert a C expression to a Spec (for simple expressions only)
    ///
    /// Returns None for complex expressions that can't be represented as Spec
    pub(crate) fn cexpr_to_spec(expr: &CExpr) -> Option<Spec> {
        match expr {
            CExpr::Var(name) => Some(Spec::var(name)),
            CExpr::IntLit(n) => Some(Spec::int(*n)),
            CExpr::BinOp { op, left, right } => {
                let l = Self::cexpr_to_spec(left)?;
                let r = Self::cexpr_to_spec(right)?;
                let spec_op = match op {
                    BinOp::Add => BinOp::Add,
                    BinOp::Sub => BinOp::Sub,
                    BinOp::Mul => BinOp::Mul,
                    BinOp::Div => BinOp::Div,
                    BinOp::Mod => BinOp::Mod,
                    _ => return None, // Other operators not supported
                };
                Some(Spec::BinOp {
                    op: spec_op,
                    left: Box::new(l),
                    right: Box::new(r),
                })
            }
            CExpr::UnaryOp { op, operand } => {
                let inner = Self::cexpr_to_spec(operand)?;
                match op {
                    UnaryOp::Neg => Some(Spec::BinOp {
                        op: BinOp::Sub,
                        left: Box::new(Spec::int(0)),
                        right: Box::new(inner),
                    }),
                    _ => None,
                }
            }
            CExpr::Index { array, index } => {
                if let CExpr::Var(arr_name) = array.as_ref() {
                    let idx = Self::cexpr_to_spec(index)?;
                    Some(Spec::Index {
                        base: Box::new(Spec::var(arr_name)),
                        index: Box::new(idx),
                    })
                } else {
                    None
                }
            }
            _ => None, // Other expressions not supported
        }
    }

    /// Helper to collect array accesses recursively
    fn collect_array_accesses(stmt: &CStmt, accesses: &mut Vec<(String, CExpr, bool)>) {
        match stmt {
            CStmt::Expr(expr) => Self::collect_array_accesses_expr(expr, accesses, false),
            CStmt::Block(stmts) => {
                for s in stmts {
                    Self::collect_array_accesses(s, accesses);
                }
            }
            CStmt::If {
                cond,
                then_stmt,
                else_stmt,
            } => {
                Self::collect_array_accesses_expr(cond, accesses, false);
                Self::collect_array_accesses(then_stmt, accesses);
                if let Some(else_s) = else_stmt {
                    Self::collect_array_accesses(else_s, accesses);
                }
            }
            CStmt::While { cond, body, .. }
            | CStmt::DoWhile { cond, body, .. }
            | CStmt::Switch { cond, body, .. } => {
                Self::collect_array_accesses_expr(cond, accesses, false);
                Self::collect_array_accesses(body, accesses);
            }
            CStmt::For {
                init,
                cond,
                update,
                body,
                ..
            } => {
                if let Some(i) = init {
                    Self::collect_array_accesses(i, accesses);
                }
                if let Some(c) = cond {
                    Self::collect_array_accesses_expr(c, accesses, false);
                }
                if let Some(u) = update {
                    Self::collect_array_accesses_expr(u, accesses, false);
                }
                Self::collect_array_accesses(body, accesses);
            }
            CStmt::Return(Some(expr)) => {
                Self::collect_array_accesses_expr(expr, accesses, false);
            }
            CStmt::Case { stmt, .. } | CStmt::Label { stmt, .. } => {
                Self::collect_array_accesses(stmt, accesses);
            }
            _ => {}
        }
    }

    /// Helper to collect array accesses from an expression
    /// `in_lhs` indicates if we're in the left-hand side of an assignment
    fn collect_array_accesses_expr(
        expr: &CExpr,
        accesses: &mut Vec<(String, CExpr, bool)>,
        in_lhs: bool,
    ) {
        match expr {
            CExpr::Index { array, index } => {
                // Extract array name
                if let CExpr::Var(arr_name) = array.as_ref() {
                    accesses.push((arr_name.clone(), *index.clone(), in_lhs));
                }
                // Also traverse index expression for nested accesses
                Self::collect_array_accesses_expr(index, accesses, false);
            }
            CExpr::BinOp { op, left, right } => {
                match op {
                    BinOp::Assign
                    | BinOp::AddAssign
                    | BinOp::SubAssign
                    | BinOp::MulAssign
                    | BinOp::DivAssign
                    | BinOp::ModAssign
                    | BinOp::BitAndAssign
                    | BinOp::BitOrAssign
                    | BinOp::BitXorAssign => {
                        // Left side is being written to
                        Self::collect_array_accesses_expr(left, accesses, true);
                        Self::collect_array_accesses_expr(right, accesses, false);
                    }
                    _ => {
                        Self::collect_array_accesses_expr(left, accesses, false);
                        Self::collect_array_accesses_expr(right, accesses, false);
                    }
                }
            }
            CExpr::UnaryOp { operand, op } => {
                // Pre/post inc/dec are also writes
                let is_write = matches!(
                    op,
                    UnaryOp::PreInc | UnaryOp::PostInc | UnaryOp::PreDec | UnaryOp::PostDec
                );
                Self::collect_array_accesses_expr(operand, accesses, is_write || in_lhs);
            }
            CExpr::Call { args, .. } => {
                for arg in args {
                    Self::collect_array_accesses_expr(arg, accesses, false);
                }
            }
            CExpr::Conditional {
                cond,
                then_expr,
                else_expr,
            } => {
                Self::collect_array_accesses_expr(cond, accesses, false);
                Self::collect_array_accesses_expr(then_expr, accesses, false);
                Self::collect_array_accesses_expr(else_expr, accesses, false);
            }
            CExpr::Cast { expr, .. } => {
                Self::collect_array_accesses_expr(expr, accesses, in_lhs);
            }
            _ => {}
        }
    }
}
