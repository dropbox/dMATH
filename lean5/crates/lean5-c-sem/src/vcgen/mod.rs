//! Verification Condition Generator for C Programs
//!
//! This module generates verification conditions (VCs) from C programs with ACSL
//! specifications. The VCs are Lean5 propositions that, if proven, establish that
//! the C code satisfies its specification.
//!
//! ## Approach: Weakest Precondition (WP) Calculus
//!
//! We use the weakest precondition approach from Dijkstra, as implemented in
//! Frama-C/WP and similar tools:
//!
//! - `wp(skip, Q) = Q`
//! - `wp(x = e, Q) = Q[e/x]`
//! - `wp(s1; s2, Q) = wp(s1, wp(s2, Q))`
//! - `wp(if b then s1 else s2, Q) = (b → wp(s1, Q)) ∧ (¬b → wp(s2, Q))`
//! - `wp(while b inv I { s }, Q) = I ∧ ∀state. (I ∧ b → wp(s, I)) ∧ (I ∧ ¬b → Q)`
//!
//! ## Example
//!
//! ```ignore
//! /*@ requires n >= 0;
//!     ensures \result >= 0;
//! */
//! int abs(int n) {
//!     if (n < 0)
//!         return -n;
//!     else
//!         return n;
//! }
//! ```
//!
//! Generates VCs:
//! 1. `n >= 0 ∧ n < 0 → -n >= 0` (negative branch)
//! 2. `n >= 0 ∧ n >= 0 → n >= 0` (positive branch)

mod assigns;
mod collect;
mod inference;
mod subst;

#[cfg(test)]
mod tests;

use crate::expr::{BinOp, CExpr, Initializer};
use crate::spec::{FuncSpec, Location, LoopSpec, Spec};
use crate::stmt::{CStmt, CaseLabel, FuncDef};
use std::collections::HashMap;

// Re-export from submodules
#[cfg(test)]
pub(crate) use inference::AccumulatorOp;
pub use inference::{GhostKind, GhostVariable, InferenceContext, InvariantInference};

/// A verification condition to be proven
#[derive(Debug, Clone)]
pub struct VC {
    /// Human-readable description of this VC
    pub description: String,
    /// The proposition to prove (as a Spec)
    pub obligation: Spec,
    /// Source location (line number, if known)
    pub location: Option<usize>,
    /// Kind of VC (precondition, postcondition, loop invariant, etc.)
    pub kind: VCKind,
}

/// Classification of verification conditions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VCKind {
    /// Function precondition must hold at call site
    Precondition,
    /// Function postcondition must hold on return
    Postcondition,
    /// Loop invariant must hold on entry
    LoopInvariantEntry,
    /// Loop invariant must be preserved by loop body
    LoopInvariantPreserved,
    /// Loop variant must decrease
    LoopVariantDecreases,
    /// Loop variant must be non-negative
    LoopVariantNonNegative,
    /// Assertion in code
    Assertion,
    /// Memory safety (valid pointer)
    MemorySafety,
    /// No undefined behavior
    NoUB,
    /// Assigns clause respected
    AssignsClause,
}

/// A modified location with metadata for error reporting
#[derive(Debug, Clone)]
pub struct ModifiedLocation {
    /// The location being modified
    pub location: Location,
    /// Human-readable description of the modification
    pub description: String,
    /// Source line number (if known)
    pub source_line: Option<usize>,
}

/// Verification condition generator
pub struct VCGen {
    /// Generated VCs
    pub(crate) vcs: Vec<VC>,
    /// Current path condition (assumptions along current path)
    pub(crate) path_condition: Vec<Spec>,
    /// Known function specifications
    pub(crate) func_specs: HashMap<String, FuncSpec>,
    /// Counter for generating fresh variable names
    pub(crate) fresh_counter: usize,
}

impl Default for VCGen {
    fn default() -> Self {
        Self::new()
    }
}

impl VCGen {
    pub fn new() -> Self {
        Self {
            vcs: Vec::new(),
            path_condition: Vec::new(),
            func_specs: HashMap::new(),
            fresh_counter: 0,
        }
    }

    /// Register a function's specification
    pub fn register_func_spec(&mut self, name: &str, spec: FuncSpec) {
        self.func_specs.insert(name.to_string(), spec);
    }

    /// Generate a fresh variable name
    pub(crate) fn fresh_var(&mut self, base: &str) -> String {
        self.fresh_counter += 1;
        format!("{}_{}", base, self.fresh_counter)
    }

    /// Add a VC to prove
    pub(crate) fn add_vc(
        &mut self,
        kind: VCKind,
        description: &str,
        obligation: Spec,
        location: Option<usize>,
    ) {
        // Incorporate path condition: path_condition → obligation
        let full_obligation = if self.path_condition.is_empty() {
            obligation
        } else {
            let path = Spec::and(self.path_condition.clone());
            Spec::implies(path, obligation)
        };

        self.vcs.push(VC {
            description: description.to_string(),
            obligation: full_obligation,
            location,
            kind,
        });
    }

    /// Generate VCs for a function
    pub fn gen_function(&mut self, func: &FuncDef, spec: &FuncSpec) -> Vec<VC> {
        self.vcs.clear();
        self.path_condition.clear();

        // Start with precondition as assumption
        for req in &spec.requires {
            self.path_condition.push(req.clone());
        }

        // Compute WP of body with postcondition
        let postcondition = if spec.ensures.is_empty() {
            Spec::True
        } else {
            Spec::and(spec.ensures.clone())
        };

        let wp = self.wp_stmt(&func.body, &postcondition, None);

        // VC: precondition → wp(body, postcondition)
        let precondition = if spec.requires.is_empty() {
            Spec::True
        } else {
            Spec::and(spec.requires.clone())
        };

        self.vcs.push(VC {
            description: format!("Function {} satisfies its contract", func.name),
            obligation: Spec::implies(precondition, wp),
            location: None,
            kind: VCKind::Postcondition,
        });

        // Check assigns clause if present
        if !spec.assigns.is_empty() {
            // Collect all modified locations in the function body
            let modified = self.collect_modified_locations(&func.body);

            // Collect local variable names (parameters + locals in body)
            let mut locals = self.collect_local_variables(&func.body);
            // Function parameters are also local to the function
            for param in &func.params {
                locals.push(param.name.clone());
            }

            // Filter out local variables - they don't affect external state
            let non_local_modified = self.filter_non_locals(modified, &locals);

            // Generate VCs for assigns clause violations
            let assigns_vcs = self.check_assigns(&spec.assigns, &non_local_modified);
            self.vcs.extend(assigns_vcs);
        }

        std::mem::take(&mut self.vcs)
    }

    /// Compute weakest precondition of a statement
    ///
    /// `wp(stmt, Q)` = weakest P such that {P} stmt {Q}
    pub fn wp_stmt(&mut self, stmt: &CStmt, postcond: &Spec, loop_spec: Option<&LoopSpec>) -> Spec {
        match stmt {
            CStmt::Empty => {
                // wp(skip, Q) = Q
                postcond.clone()
            }

            CStmt::Expr(e) => {
                // For expression statements, check for side effects
                self.wp_expr(e, postcond)
            }

            CStmt::Decl(decl) => {
                // wp(T x = e, Q) = Q[e/x] if initialized, otherwise Q
                if let Some(Initializer::Expr(init)) = &decl.init {
                    self.substitute(postcond, &decl.name, init)
                } else {
                    postcond.clone()
                }
            }

            CStmt::Block(stmts) => {
                // wp(s1; s2; ...; sn, Q) = wp(s1, wp(s2, ... wp(sn, Q)))
                let mut q = postcond.clone();
                for s in stmts.iter().rev() {
                    q = self.wp_stmt(s, &q, loop_spec);
                }
                q
            }

            CStmt::If {
                cond,
                then_stmt,
                else_stmt,
            } => {
                // wp(if b then s1 else s2, Q) = (b → wp(s1, Q)) ∧ (¬b → wp(s2, Q))
                let cond_spec = self.expr_to_spec(cond);
                let wp_then = self.wp_stmt(then_stmt, postcond, loop_spec);

                let wp_else = if let Some(else_s) = else_stmt {
                    self.wp_stmt(else_s, postcond, loop_spec)
                } else {
                    postcond.clone()
                };

                Spec::and(vec![
                    Spec::implies(cond_spec.clone(), wp_then),
                    Spec::implies(Spec::not(cond_spec), wp_else),
                ])
            }

            CStmt::While { cond, body } => self.wp_while(cond, body, postcond, loop_spec),

            CStmt::DoWhile { cond, body } => {
                // do { body } while (cond) ≡ body; while (cond) { body }
                let while_wp = self.wp_while(cond, body, postcond, loop_spec);
                self.wp_stmt(body, &while_wp, loop_spec)
            }

            CStmt::For {
                init,
                cond,
                update,
                body,
            } => {
                // for (init; cond; update) body ≡ init; while (cond) { body; update }
                let cond_expr = cond.clone().unwrap_or_else(|| CExpr::int(1)); // Missing cond = true

                // Build equivalent while loop body: body; update
                let while_body = if let Some(upd) = update {
                    CStmt::Block(vec![(**body).clone(), CStmt::Expr(upd.clone())])
                } else {
                    (**body).clone()
                };

                let while_wp = self.wp_while(&cond_expr, &while_body, postcond, loop_spec);

                // Apply init
                if let Some(init_stmt) = init {
                    self.wp_stmt(init_stmt, &while_wp, loop_spec)
                } else {
                    while_wp
                }
            }

            CStmt::Return(expr) => {
                // wp(return e, Q) = Q[\result ← e]
                if let Some(e) = expr {
                    self.substitute_result(postcond, e)
                } else {
                    postcond.clone()
                }
            }

            CStmt::Break => {
                // Break transfers control outside the loop
                // In WP calculus, we need the loop's postcondition
                // This is handled by loop_spec if available
                postcond.clone()
            }

            CStmt::Continue => {
                // Continue jumps to loop condition check
                // Need loop invariant
                if let Some(ls) = loop_spec {
                    if ls.invariant.is_empty() {
                        Spec::True
                    } else {
                        Spec::and(ls.invariant.clone())
                    }
                } else {
                    postcond.clone()
                }
            }

            CStmt::Switch { cond, body } => {
                // wp(switch(e) { case c1: s1; ... case cn: sn; default: sd; }, Q)
                //
                // For proper switch semantics, we need to handle:
                // 1. Case matching: which case is selected
                // 2. Fallthrough: cases without break continue to next case
                // 3. Default case: executed if no case matches
                //
                // We transform switch into equivalent if-else chain:
                // if (e == c1) { s1... } else if (e == c2) { s2... } else { sd }
                //
                // For fallthrough (no break), the statements of multiple cases are combined.

                let cond_spec = self.expr_to_spec(cond);

                // Extract cases from switch body
                let cases = self.extract_switch_cases(body);

                if cases.is_empty() {
                    // Empty switch - just evaluate condition for side effects
                    return postcond.clone();
                }

                // Build if-else chain from cases
                let mut has_default = false;
                let mut default_wp = postcond.clone();

                // Find default case first
                for (label, stmt_body) in &cases {
                    if matches!(label, CaseLabel::Default) {
                        has_default = true;
                        default_wp = self.wp_stmt(stmt_body, postcond, loop_spec);
                        break;
                    }
                }

                // Build conditions for each case
                let mut case_conditions = Vec::new();

                for (label, stmt_body) in &cases {
                    if let CaseLabel::Case(case_expr) = label {
                        let case_spec = self.expr_to_spec(case_expr);
                        let case_cond = Spec::binop(BinOp::Eq, cond_spec.clone(), case_spec);
                        let case_wp = self.wp_stmt(stmt_body, postcond, loop_spec);
                        case_conditions.push((case_cond, case_wp));
                    }
                }

                if case_conditions.is_empty() && has_default {
                    // Only default case - always execute it
                    return default_wp;
                }

                // Build: (c1 → wp1) ∧ (c2 → wp2) ∧ ... ∧ (¬c1 ∧ ¬c2 ∧ ... → default_wp)
                let mut conjuncts = Vec::new();

                // Add implications for each case
                for (case_cond, case_wp) in &case_conditions {
                    conjuncts.push(Spec::implies(case_cond.clone(), case_wp.clone()));
                }

                // Add default/no-match case
                if !case_conditions.is_empty() {
                    let no_match_cond = Spec::and(
                        case_conditions
                            .iter()
                            .map(|(c, _)| Spec::not(c.clone()))
                            .collect(),
                    );

                    if has_default {
                        conjuncts.push(Spec::implies(no_match_cond, default_wp));
                    } else {
                        // No default - if no case matches, just postcond holds
                        conjuncts.push(Spec::implies(no_match_cond, postcond.clone()));
                    }
                }

                Spec::and(conjuncts)
            }

            CStmt::Goto(_) | CStmt::Label { .. } => {
                // Goto requires more sophisticated analysis
                postcond.clone()
            }

            CStmt::Assert(spec) => {
                // wp(assert P, Q) = P ∧ Q
                self.add_vc(VCKind::Assertion, "Assertion must hold", spec.clone(), None);
                Spec::and(vec![spec.clone(), postcond.clone()])
            }

            CStmt::Assume(spec) => {
                // wp(assume P, Q) = P → Q
                Spec::implies(spec.clone(), postcond.clone())
            }

            CStmt::DeclList(decls) => {
                // Handle multiple declarations by processing in reverse
                let mut q = postcond.clone();
                for decl in decls.iter().rev() {
                    if let Some(Initializer::Expr(init)) = &decl.init {
                        q = self.substitute(&q, &decl.name, init);
                    }
                }
                q
            }

            CStmt::Case { stmt, .. } => {
                // Switch case - just process the inner statement
                self.wp_stmt(stmt, postcond, loop_spec)
            }

            CStmt::FuncDef(_) => {
                // Function definition inside statement - ignore for WP
                postcond.clone()
            }

            CStmt::Asm(_) => {
                // Inline assembly - can't reason about it
                postcond.clone()
            }
        }
    }

    /// WP for while loops
    fn wp_while(
        &mut self,
        cond: &CExpr,
        body: &CStmt,
        postcond: &Spec,
        loop_spec: Option<&LoopSpec>,
    ) -> Spec {
        let cond_spec = self.expr_to_spec(cond);

        // If we have a loop specification, use it
        if let Some(ls) = loop_spec {
            let invariant = if ls.invariant.is_empty() {
                Spec::True
            } else {
                Spec::and(ls.invariant.clone())
            };

            // Generate VCs for loop:

            // 1. Loop invariant holds on entry (this is a precondition to the loop)
            // Already implied by requiring I in the WP

            // 2. Loop body preserves invariant: I ∧ cond → wp(body, I)
            let wp_body = self.wp_stmt(body, &invariant, Some(ls));
            self.add_vc(
                VCKind::LoopInvariantPreserved,
                "Loop body preserves invariant",
                Spec::implies(
                    Spec::and(vec![invariant.clone(), cond_spec.clone()]),
                    wp_body,
                ),
                None,
            );

            // 3. Invariant + ¬cond → postcondition
            self.add_vc(
                VCKind::Postcondition,
                "Loop exit satisfies postcondition",
                Spec::implies(
                    Spec::and(vec![invariant.clone(), Spec::not(cond_spec.clone())]),
                    postcond.clone(),
                ),
                None,
            );

            // 4. If there's a variant, it decreases and stays non-negative
            if let Some(variant) = &ls.variant {
                let variant_var = self.fresh_var("variant");
                let old_variant = Spec::Let {
                    var: variant_var.clone(),
                    value: Box::new(variant.clone()),
                    body: Box::new(Spec::True),
                };

                // Variant decreases
                self.add_vc(
                    VCKind::LoopVariantDecreases,
                    "Loop variant decreases",
                    Spec::implies(
                        Spec::and(vec![invariant.clone(), cond_spec.clone()]),
                        Spec::lt(variant.clone(), Spec::var(&variant_var)),
                    ),
                    None,
                );

                // Variant is non-negative
                self.add_vc(
                    VCKind::LoopVariantNonNegative,
                    "Loop variant is non-negative",
                    Spec::implies(
                        Spec::and(vec![invariant.clone(), cond_spec.clone()]),
                        Spec::ge(variant.clone(), Spec::int(0)),
                    ),
                    None,
                );

                let _ = old_variant; // Suppress unused warning
            }

            // WP of loop is the invariant (caller must establish it)
            invariant
        } else {
            // No explicit loop spec - try automatic invariant inference
            let mut inference = InvariantInference::new();
            let mut context = InferenceContext::new();

            // Extract loop counter variable from condition (if present)
            let loop_var = Self::extract_loop_variable(cond);

            // Extract modified variables from body for ghost tracking
            let modified_vars = Self::extract_modified_variables(body);

            // Generate ghost variables for the loop
            context.generate_loop_ghosts(loop_var.as_deref(), &modified_vars);

            // Extract array accesses and generate frame ghosts for read-only elements
            let array_accesses = Self::extract_array_accesses(body);
            // Convert CExpr indices to Spec for ghost tracking
            let spec_accesses: Vec<(String, Spec, bool)> = array_accesses
                .iter()
                .filter_map(|(arr, idx, is_write)| {
                    Self::cexpr_to_spec(idx).map(|spec_idx| (arr.clone(), spec_idx, *is_write))
                })
                .collect();
            context.generate_array_frame_ghosts(&spec_accesses);

            // Collect inferred invariants
            let mut inferred_invariants = inference.infer_while_invariant(cond, body, &context);

            // Also try search pattern detection
            if let Some(search_inv) = inference.detect_search_pattern(cond, body) {
                inferred_invariants.push(search_inv);
            }

            if inferred_invariants.is_empty() {
                // Could not infer any invariant - generate warning VC
                self.add_vc(
                    VCKind::LoopInvariantEntry,
                    "Loop requires invariant annotation (automatic inference failed)",
                    Spec::True, // Can't verify without invariant
                    None,
                );
                postcond.clone()
            } else {
                // Use the inferred invariants
                let invariant = if inferred_invariants.len() == 1 {
                    inferred_invariants.pop().unwrap()
                } else {
                    Spec::and(inferred_invariants)
                };

                // Generate VCs for inferred invariant
                self.add_vc(
                    VCKind::LoopInvariantEntry,
                    "Inferred invariant holds on entry",
                    invariant.clone(),
                    None,
                );

                // Loop body preserves invariant: I ∧ cond → wp(body, I)
                let wp_body = self.wp_stmt(body, &invariant, None);
                self.add_vc(
                    VCKind::LoopInvariantPreserved,
                    "Inferred invariant preserved by loop body",
                    Spec::implies(
                        Spec::and(vec![invariant.clone(), cond_spec.clone()]),
                        wp_body,
                    ),
                    None,
                );

                // Invariant + ¬cond → postcondition
                self.add_vc(
                    VCKind::Postcondition,
                    "Loop exit with inferred invariant satisfies postcondition",
                    Spec::implies(
                        Spec::and(vec![invariant.clone(), Spec::not(cond_spec.clone())]),
                        postcond.clone(),
                    ),
                    None,
                );

                invariant
            }
        }
    }

    /// Compute WP for expressions with side effects
    fn wp_expr(&mut self, expr: &CExpr, postcond: &Spec) -> Spec {
        match expr {
            CExpr::BinOp {
                op: BinOp::Assign,
                left,
                right,
            } => {
                // wp(x = e, Q) = Q[e/x]
                if let CExpr::Var(name) = left.as_ref() {
                    self.substitute(postcond, name, right)
                } else {
                    // Complex LHS (e.g., *p = e, a[i] = e)
                    // Would need memory model reasoning
                    postcond.clone()
                }
            }

            CExpr::UnaryOp { op, operand } => {
                match op {
                    crate::expr::UnaryOp::PreInc | crate::expr::UnaryOp::PostInc => {
                        if let CExpr::Var(name) = operand.as_ref() {
                            // x++ or ++x: Q[x+1/x]
                            let incremented = CExpr::add(CExpr::var(name), CExpr::int(1));
                            self.substitute(postcond, name, &incremented)
                        } else {
                            postcond.clone()
                        }
                    }
                    crate::expr::UnaryOp::PreDec | crate::expr::UnaryOp::PostDec => {
                        if let CExpr::Var(name) = operand.as_ref() {
                            let decremented = CExpr::sub(CExpr::var(name), CExpr::int(1));
                            self.substitute(postcond, name, &decremented)
                        } else {
                            postcond.clone()
                        }
                    }
                    crate::expr::UnaryOp::Deref => {
                        // Check pointer validity
                        let ptr_spec = self.expr_to_spec(operand);
                        self.add_vc(
                            VCKind::MemorySafety,
                            "Pointer dereference is valid",
                            Spec::valid(ptr_spec),
                            None,
                        );
                        postcond.clone()
                    }
                    _ => postcond.clone(),
                }
            }

            CExpr::Call { func, args } => {
                // Function call: check precondition, assume postcondition
                if let CExpr::Var(func_name) = func.as_ref() {
                    if let Some(func_spec) = self.func_specs.get(func_name).cloned() {
                        // Convert actual arguments to Spec for substitution
                        let arg_specs: Vec<Spec> =
                            args.iter().map(|a| self.expr_to_spec(a)).collect();

                        // Add VC for precondition (substituting actual args for formals)
                        for req in &func_spec.requires {
                            // Substitute formal parameters with actual arguments
                            let subst_req = self.subst_params(req, &func_spec.params, &arg_specs);
                            self.add_vc(
                                VCKind::Precondition,
                                &format!("Precondition of {func_name} must hold"),
                                subst_req,
                                None,
                            );
                        }
                        // Assume postcondition holds for the result value
                        // WP: (precondition → postcondition[\result ← call]) → Q
                        // Simplified: we assume the postcondition is available as a hypothesis
                        if !func_spec.ensures.is_empty() {
                            // The callee's postcondition becomes available as an assumption
                            // For the result, we substitute \result with the call expression
                            let call_spec = Spec::Call {
                                func: func_name.clone(),
                                args: arg_specs.clone(),
                            };
                            let callee_postcond = Spec::and(func_spec.ensures.clone());
                            // First substitute formal params with actual args
                            let param_subst =
                                self.subst_params(&callee_postcond, &func_spec.params, &arg_specs);
                            // Resolve \old() expressions: callee's \old(param) becomes actual arg value at call site
                            let old_resolved = self.resolve_old_for_call(&param_subst);
                            // Then substitute \result with the call
                            let instantiated = self.subst_result(&old_resolved, &call_spec);
                            // Return: callee_postcond → Q
                            // This means: assuming callee's postcondition, we need Q
                            return Spec::implies(instantiated, postcond.clone());
                        }
                    }
                }
                postcond.clone()
            }

            CExpr::BinOp {
                op: BinOp::Div | BinOp::Mod,
                left: _,
                right,
            } => {
                // Check division by zero
                let divisor = self.expr_to_spec(right);
                self.add_vc(
                    VCKind::NoUB,
                    "Divisor is non-zero",
                    Spec::ne(divisor, Spec::int(0)),
                    None,
                );
                postcond.clone()
            }

            CExpr::Index { array, index } => {
                // Array bounds check (if we know bounds)
                let arr_spec = self.expr_to_spec(array);
                let idx_spec = self.expr_to_spec(index);

                // Check index >= 0
                self.add_vc(
                    VCKind::MemorySafety,
                    "Array index is non-negative",
                    Spec::ge(idx_spec.clone(), Spec::int(0)),
                    None,
                );

                // Check index < length (need array length info)
                // For now, generate a validity check
                self.add_vc(
                    VCKind::MemorySafety,
                    "Array access is within bounds",
                    Spec::ValidRange {
                        ptr: Box::new(arr_spec),
                        lo: Box::new(Spec::int(0)),
                        hi: Box::new(idx_spec),
                    },
                    None,
                );

                postcond.clone()
            }

            _ => postcond.clone(),
        }
    }

    /// Convert a C expression to a Spec for use in logical reasoning
    pub(crate) fn expr_to_spec(&self, expr: &CExpr) -> Spec {
        match expr {
            CExpr::IntLit(n) => Spec::Int(*n),
            CExpr::Var(name) => Spec::Var(name.clone()),
            CExpr::BinOp { op, left, right } => {
                let l = self.expr_to_spec(left);
                let r = self.expr_to_spec(right);
                Spec::binop(*op, l, r)
            }
            CExpr::UnaryOp { op, operand } => Spec::UnaryOp {
                op: *op,
                operand: Box::new(self.expr_to_spec(operand)),
            },
            _ => Spec::Expr(expr.clone()),
        }
    }

    /// Get the generated VCs
    pub fn get_vcs(&self) -> &[VC] {
        &self.vcs
    }
}

/// Convert VCs to Lean5 kernel expressions for proving
pub fn vc_to_lean5(vc: &VC) -> lean5_kernel::Expr {
    let mut ctx = crate::translate::TranslationContext::new();
    ctx.translate_spec(&vc.obligation)
}
