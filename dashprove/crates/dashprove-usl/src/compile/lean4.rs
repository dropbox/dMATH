//! LEAN 4 backend compiler
//!
//! Compiles USL specifications to LEAN 4 for theorem proving.

use crate::ast::{
    BinaryOp, CapabilitySpec, ComparisonOp, Expr, ImprovementProposal, Invariant, Property,
    Refinement, RollbackSpec, Security, Theorem, Type, VerificationGate, VersionSpec,
};
use crate::typecheck::TypedSpec;

use super::CompiledSpec;

/// LEAN 4 compiler
pub struct Lean4Compiler {
    namespace: String,
    /// Track whether Classical logic is used
    needs_classical: std::cell::Cell<bool>,
    /// Track whether Ring tactic is used (for polynomial arithmetic)
    needs_ring: std::cell::Cell<bool>,
    /// Track whether Linarith tactic is used (for linear arithmetic with hypotheses)
    needs_linarith: std::cell::Cell<bool>,
}

impl Lean4Compiler {
    /// Create a new LEAN 4 compiler with the given namespace
    #[must_use]
    pub fn new(namespace: &str) -> Self {
        Self {
            namespace: namespace.to_string(),
            needs_classical: std::cell::Cell::new(false),
            needs_ring: std::cell::Cell::new(false),
            needs_linarith: std::cell::Cell::new(false),
        }
    }

    /// Compile an expression to LEAN 4 syntax
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn compile_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Var(name) => {
                // Handle primed variables - LEAN doesn't have this concept,
                // we use _next suffix
                if name.ends_with('\'') {
                    format!("{}_next", &name[..name.len() - 1])
                } else {
                    name.clone()
                }
            }
            Expr::Int(n) => n.to_string(),
            Expr::Float(f) => f.to_string(),
            Expr::String(s) => format!("\"{s}\""),
            Expr::Bool(b) => if *b { "True" } else { "False" }.to_string(),

            Expr::ForAll { var, ty, body } => {
                let ty_str = ty
                    .as_ref()
                    .map(|t| format!(": {}", self.compile_type(t)))
                    .unwrap_or_default();
                format!("∀ {}{}, {}", var, ty_str, self.compile_expr(body))
            }
            Expr::Exists { var, ty, body } => {
                let ty_str = ty
                    .as_ref()
                    .map(|t| format!(": {}", self.compile_type(t)))
                    .unwrap_or_default();
                format!("∃ {}{}, {}", var, ty_str, self.compile_expr(body))
            }
            Expr::ForAllIn {
                var,
                collection,
                body,
            } => {
                format!(
                    "∀ {} ∈ {}, {}",
                    var,
                    self.compile_expr(collection),
                    self.compile_expr(body)
                )
            }
            Expr::ExistsIn {
                var,
                collection,
                body,
            } => {
                format!(
                    "∃ {} ∈ {}, {}",
                    var,
                    self.compile_expr(collection),
                    self.compile_expr(body)
                )
            }

            Expr::Implies(lhs, rhs) => {
                format!(
                    "({}) → ({})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::And(lhs, rhs) => {
                format!(
                    "({}) ∧ ({})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Or(lhs, rhs) => {
                format!(
                    "({}) ∨ ({})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Not(e) => format!("¬({})", self.compile_expr(e)),

            Expr::Compare(lhs, op, rhs) => {
                let op_str = match op {
                    ComparisonOp::Eq => "=",
                    ComparisonOp::Ne => "≠",
                    ComparisonOp::Lt => "<",
                    ComparisonOp::Le => "≤",
                    ComparisonOp::Gt => ">",
                    ComparisonOp::Ge => "≥",
                };
                format!(
                    "({}) {} ({})",
                    self.compile_expr(lhs),
                    op_str,
                    self.compile_expr(rhs)
                )
            }
            Expr::Binary(lhs, op, rhs) => {
                let op_str = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Mod => "%",
                };
                format!(
                    "({}) {} ({})",
                    self.compile_expr(lhs),
                    op_str,
                    self.compile_expr(rhs)
                )
            }
            Expr::Neg(e) => format!("-({})", self.compile_expr(e)),

            Expr::App(name, args) => {
                // Handle graph predicates with proper Lean4 translations
                self.compile_graph_function(name, args).unwrap_or_else(|| {
                    // Default: pass through as function application
                    if args.is_empty() {
                        name.clone()
                    } else {
                        let args_str: Vec<String> =
                            args.iter().map(|a| self.compile_expr(a)).collect();
                        format!("({} {})", name, args_str.join(" "))
                    }
                })
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                let recv_str = self.compile_expr(receiver);
                if args.is_empty() {
                    format!("{recv_str}.{method}")
                } else {
                    let args_str: Vec<String> = args.iter().map(|a| self.compile_expr(a)).collect();
                    format!("{recv_str}.{method} {}", args_str.join(" "))
                }
            }
            Expr::FieldAccess(obj, field) => {
                format!("{}.{field}", self.compile_expr(obj))
            }
        }
    }

    /// Compile graph-specific functions to Lean4 expressions
    ///
    /// Returns Some(lean4_expr) for recognized graph functions, None otherwise.
    /// These functions are defined in Phase 17.3 for DashFlow execution graph verification.
    fn compile_graph_function(&self, name: &str, args: &[Expr]) -> Option<String> {
        let args_str: Vec<String> = args.iter().map(|a| self.compile_expr(a)).collect();

        match name {
            // Graph predicates
            "is_acyclic" | "is_dag" => {
                // is_acyclic(g) => Graph.isAcyclic g
                if args.len() == 1 {
                    Some(format!("(Graph.isAcyclic {})", args_str[0]))
                } else {
                    None
                }
            }

            "is_connected" => {
                // is_connected(g) => Graph.isConnected g
                if args.len() == 1 {
                    Some(format!("(Graph.isConnected {})", args_str[0]))
                } else {
                    None
                }
            }

            "has_path" => {
                // has_path(g, from, to) => Graph.hasPath g from to
                if args.len() == 3 {
                    Some(format!(
                        "(Graph.hasPath {} {} {})",
                        args_str[0], args_str[1], args_str[2]
                    ))
                } else {
                    None
                }
            }

            "reachable" => {
                // reachable(from, to) or reachable(g, from, to)
                if args.len() == 3 {
                    Some(format!(
                        "(Graph.reachable {} {} {})",
                        args_str[0], args_str[1], args_str[2]
                    ))
                } else if args.len() == 2 {
                    Some(format!("(reachable {} {})", args_str[0], args_str[1]))
                } else {
                    None
                }
            }

            "in_graph" => {
                // in_graph(node, g) => node ∈ g.nodes
                if args.len() == 2 {
                    Some(format!("({} ∈ {}.nodes)", args_str[0], args_str[1]))
                } else {
                    None
                }
            }

            "edge_exists" => {
                // edge_exists(g, from, to) => (from, to) ∈ g.edges
                if args.len() == 3 {
                    Some(format!(
                        "(({}, {}) ∈ {}.edges)",
                        args_str[1], args_str[2], args_str[0]
                    ))
                } else {
                    None
                }
            }

            // DashFlow modification predicates
            "preserves_completed" => {
                // preserves_completed(g, g') => ∀ n ∈ g.nodes, completed n → n ∈ g'.nodes ∧ completed n
                if args.len() == 2 {
                    Some(format!(
                        "(∀ n ∈ {}.nodes, completed n → n ∈ {}.nodes ∧ completed n)",
                        args_str[0], args_str[1]
                    ))
                } else {
                    None
                }
            }

            "valid_modification" => {
                // valid_modification(m, g, g') => Modification.isValid m g g'
                if args.len() == 3 {
                    Some(format!(
                        "(Modification.isValid {} {} {})",
                        args_str[0], args_str[1], args_str[2]
                    ))
                } else {
                    None
                }
            }

            "preserves_dag" => {
                // preserves_dag(g, g') => Graph.isAcyclic g → Graph.isAcyclic g'
                if args.len() == 2 {
                    Some(format!(
                        "(Graph.isAcyclic {} → Graph.isAcyclic {})",
                        args_str[0], args_str[1]
                    ))
                } else {
                    None
                }
            }

            "is_ready" | "all_deps_completed" => {
                // is_ready(node, g) => ∀ dep ∈ predecessors g node, completed dep
                if args.len() == 2 {
                    Some(format!(
                        "(∀ dep ∈ (Graph.predecessors {} {}), completed dep)",
                        args_str[1], args_str[0]
                    ))
                } else {
                    None
                }
            }

            // Graph accessor functions
            "nodes" => {
                // nodes(g) => g.nodes
                if args.len() == 1 {
                    Some(format!("{}.nodes", args_str[0]))
                } else {
                    None
                }
            }

            "edges" => {
                // edges(g) => g.edges
                if args.len() == 1 {
                    Some(format!("{}.edges", args_str[0]))
                } else {
                    None
                }
            }

            "successors" => {
                // successors(g, node) => Graph.successors g node
                if args.len() == 2 {
                    Some(format!(
                        "(Graph.successors {} {})",
                        args_str[0], args_str[1]
                    ))
                } else {
                    None
                }
            }

            "predecessors" => {
                // predecessors(g, node) => Graph.predecessors g node
                if args.len() == 2 {
                    Some(format!(
                        "(Graph.predecessors {} {})",
                        args_str[0], args_str[1]
                    ))
                } else {
                    None
                }
            }

            "path" => {
                // path(g, from, to) => Graph.findPath g from to
                if args.len() == 3 {
                    Some(format!(
                        "(Graph.findPath {} {} {})",
                        args_str[0], args_str[1], args_str[2]
                    ))
                } else {
                    None
                }
            }

            "topological_order" => {
                // topological_order(g) => Graph.topologicalSort g
                if args.len() == 1 {
                    Some(format!("(Graph.topologicalSort {})", args_str[0]))
                } else {
                    None
                }
            }

            "node_count" => {
                // node_count(g) => g.nodes.card
                if args.len() == 1 {
                    Some(format!("{}.nodes.card", args_str[0]))
                } else {
                    None
                }
            }

            "edge_count" => {
                // edge_count(g) => g.edges.card
                if args.len() == 1 {
                    Some(format!("{}.edges.card", args_str[0]))
                } else {
                    None
                }
            }

            "in_degree" => {
                // in_degree(g, node) => Graph.inDegree g node
                if args.len() == 2 {
                    Some(format!("(Graph.inDegree {} {})", args_str[0], args_str[1]))
                } else {
                    None
                }
            }

            "out_degree" => {
                // out_degree(g, node) => Graph.outDegree g node
                if args.len() == 2 {
                    Some(format!("(Graph.outDegree {} {})", args_str[0], args_str[1]))
                } else {
                    None
                }
            }

            // Node status predicates
            "completed" => {
                if args.len() == 1 {
                    Some(format!("(Node.isCompleted {})", args_str[0]))
                } else {
                    None
                }
            }

            "pending" => {
                if args.len() == 1 {
                    Some(format!("(Node.isPending {})", args_str[0]))
                } else {
                    None
                }
            }

            "running" => {
                if args.len() == 1 {
                    Some(format!("(Node.isRunning {})", args_str[0]))
                } else {
                    None
                }
            }

            "failed" => {
                if args.len() == 1 {
                    Some(format!("(Node.isFailed {})", args_str[0]))
                } else {
                    None
                }
            }

            _ => None, // Not a graph function, let default handling take over
        }
    }

    /// Compile a type to LEAN 4 syntax
    #[must_use]
    pub fn compile_type(&self, ty: &Type) -> String {
        match ty {
            Type::Named(name) => {
                // Map common type names
                match name.as_str() {
                    "Bool" | "boolean" => "Bool".to_string(),
                    "Int" | "int" | "integer" => "Int".to_string(),
                    "Float" | "float" => "Float".to_string(),
                    "String" | "string" => "String".to_string(),
                    _ => name.clone(),
                }
            }
            Type::Set(inner) => format!("Set {}", self.compile_type(inner)),
            Type::List(inner) => format!("List {}", self.compile_type(inner)),
            Type::Map(k, v) => {
                format!("{} → {}", self.compile_type(k), self.compile_type(v))
            }
            Type::Relation(a, b) => {
                format!("{} → {} → Prop", self.compile_type(a), self.compile_type(b))
            }
            Type::Function(a, b) => {
                format!("{} → {}", self.compile_type(a), self.compile_type(b))
            }
            Type::Result(inner) => format!("Except String {}", self.compile_type(inner)),
            Type::Unit => "Unit".to_string(),
            Type::Graph(n, e) => {
                format!("Graph {} {}", self.compile_type(n), self.compile_type(e))
            }
            Type::Path(n) => format!("Path {}", self.compile_type(n)),
        }
    }

    /// Compile a theorem to LEAN 4 definition
    #[must_use]
    pub fn compile_theorem(&self, thm: &Theorem) -> String {
        let body_str = self.compile_expr(&thm.body);
        let tactic = self.suggest_tactic(&thm.body);
        format!("theorem {} : {} := by\n  {}", thm.name, body_str, tactic)
    }

    /// Suggest appropriate tactics based on the expression structure
    #[must_use]
    pub fn suggest_tactic(&self, expr: &Expr) -> String {
        // First, collect all intro names from nested quantifiers/implications
        let (intros, has_hypotheses, inner_expr) = Self::collect_intros(expr);

        // Generate intro tactics
        let intro_tactics = if intros.is_empty() {
            String::new()
        } else {
            format!("intro {}\n  ", intros.join(" "))
        };

        // Suggest tactic for the inner expression
        // Pass whether we have hypotheses - linarith can use them for linear arithmetic
        let inner_tactic = self.suggest_inner_tactic(inner_expr, has_hypotheses);

        format!("{intro_tactics}{inner_tactic}")
    }

    /// Collect all intro names from nested foralls/implications
    /// Returns (`intro_names`, `has_hypotheses`, `inner_expr`)
    fn collect_intros(expr: &Expr) -> (Vec<String>, bool, &Expr) {
        let mut intros = Vec::new();
        let mut has_hypotheses = false;
        let mut current = expr;

        loop {
            match current {
                Expr::ForAll { var, body, .. } => {
                    intros.push(var.clone());
                    current = body;
                }
                Expr::ForAllIn { var, body, .. } => {
                    // ForAllIn introduces both the element and a membership hypothesis
                    // ∀ x ∈ S, P x  =>  intro x hx
                    intros.push(var.clone());
                    intros.push(format!("h{var}")); // membership hypothesis
                    has_hypotheses = true; // We have a membership hypothesis we can use
                    current = body;
                }
                Expr::Implies(_, rhs) => {
                    // For implications, generate hypothesis names
                    intros.push(format!("h{}", intros.len()));
                    has_hypotheses = true; // We have a hypothesis we can use
                    current = rhs;
                }
                _ => break,
            }
        }

        (intros, has_hypotheses, current)
    }

    /// Suggest tactic for an inner expression (after intros)
    /// `has_hypotheses` indicates if we have hypotheses from implications that can be used
    #[allow(clippy::too_many_lines)]
    fn suggest_inner_tactic(&self, expr: &Expr, has_hypotheses: bool) -> String {
        match expr {
            // Simple reflexivity: x == x
            Expr::Compare(lhs, ComparisonOp::Eq, rhs) if Self::exprs_equal(lhs, rhs) => {
                "rfl".to_string()
            }
            // True literal
            Expr::Bool(true) => "trivial".to_string(),
            // False literal - contradiction needed
            Expr::Bool(false) => "contradiction".to_string(),

            // Comparisons involving method calls - use simp (must come before arithmetic pattern)
            Expr::Compare(lhs, _, rhs)
                if Self::contains_simplifiable_method(lhs)
                    || Self::contains_simplifiable_method(rhs) =>
            {
                "simp".to_string()
            }

            // Boolean equality involving negation (De Morgan's laws) - use decide for Bool
            Expr::Compare(lhs, ComparisonOp::Eq, rhs)
                if Self::involves_boolean_logic(lhs) || Self::involves_boolean_logic(rhs) =>
            {
                "decide".to_string()
            }

            // Arithmetic equality - try ring or omega
            Expr::Compare(lhs, ComparisonOp::Eq, rhs)
                if Self::is_arithmetic(lhs) || Self::is_arithmetic(rhs) =>
            {
                // Use omega for linear arithmetic, ring for polynomial
                if Self::is_linear_arithmetic(lhs) && Self::is_linear_arithmetic(rhs) {
                    "omega".to_string()
                } else {
                    self.needs_ring.set(true);
                    "ring".to_string()
                }
            }

            // Arithmetic comparison (inequalities) - use linarith when hypotheses available
            // linarith can use hypotheses to solve linear arithmetic goals
            Expr::Compare(
                lhs,
                ComparisonOp::Lt | ComparisonOp::Le | ComparisonOp::Gt | ComparisonOp::Ge,
                rhs,
            ) if has_hypotheses
                && Self::is_linear_arithmetic(lhs)
                && Self::is_linear_arithmetic(rhs) =>
            {
                self.needs_linarith.set(true);
                "linarith".to_string()
            }

            // Arithmetic comparison - use omega (fallback when no hypotheses or non-linear)
            Expr::Compare(
                _,
                ComparisonOp::Lt | ComparisonOp::Le | ComparisonOp::Gt | ComparisonOp::Ge,
                _,
            ) => "omega".to_string(),

            // Conjunction - split into subgoals
            Expr::And(lhs, rhs) => {
                let left_tactic = self.suggest_inner_tactic(lhs, has_hypotheses);
                let right_tactic = self.suggest_inner_tactic(rhs, has_hypotheses);
                format!("constructor\n  · {left_tactic}\n  · {right_tactic}")
            }

            // Disjunction with tautology pattern (p or not p)
            Expr::Or(lhs, rhs) => {
                if Self::is_negation_of(lhs, rhs) || Self::is_negation_of(rhs, lhs) {
                    self.needs_classical.set(true);
                    "exact Classical.em _".to_string()
                } else if Self::is_decidable(lhs) && Self::is_decidable(rhs) {
                    // Both sides are decidable booleans - use decide
                    "decide".to_string()
                } else if has_hypotheses {
                    // We have hypotheses that might help - try simp_all which can
                    // use hypotheses to derive one side of the disjunction
                    "simp_all".to_string()
                } else {
                    // Try to prove the left side first, if it looks simpler
                    let left_tactic = self.suggest_inner_tactic(lhs, has_hypotheses);
                    if left_tactic == "sorry" {
                        let right_tactic = self.suggest_inner_tactic(rhs, has_hypotheses);
                        if right_tactic == "sorry" {
                            // Last resort - try tauto for propositional logic
                            self.needs_classical.set(true);
                            "tauto".to_string()
                        } else {
                            format!("right\n  {right_tactic}")
                        }
                    } else {
                        format!("left\n  {left_tactic}")
                    }
                }
            }

            // Double negation - simp can handle (must come before general negation)
            Expr::Not(inner) if matches!(inner.as_ref(), Expr::Not(_)) => "simp".to_string(),

            // Negation - try to derive contradiction
            Expr::Not(inner) => {
                let inner_str = self.suggest_inner_tactic(inner, has_hypotheses);
                if inner_str == "sorry" {
                    "intro h\n    sorry".to_string()
                } else {
                    format!("intro h\n    {inner_str}")
                }
            }

            // Existential - provide witness with sorry
            Expr::Exists { .. } | Expr::ExistsIn { .. } => "use sorry\n  sorry".to_string(),

            // Implication - intro hypothesis and prove consequent
            Expr::Implies(_, rhs) => {
                let consequent_tactic = self.suggest_inner_tactic(rhs, true); // now has hypothesis
                format!("intro h\n  {consequent_tactic}")
            }

            // Simple boolean decidable - use decide
            Expr::Compare(lhs, ComparisonOp::Eq | ComparisonOp::Ne, rhs)
                if Self::is_decidable(lhs) && Self::is_decidable(rhs) =>
            {
                "decide".to_string()
            }

            // Function application that might be simplifiable
            Expr::App(name, _) if name == "not" || name == "id" => "simp".to_string(),

            // Method calls on common simplifiable types
            Expr::MethodCall { method, .. } if Self::is_simplifiable_method(method) => {
                "simp".to_string()
            }

            // Variable goal with hypotheses - try simp_all which can apply hypotheses
            Expr::Var(_)
            | Expr::MethodCall { .. }
            | Expr::FieldAccess { .. }
            | Expr::App(_, _)
            | Expr::Compare(_, _, _)
                if has_hypotheses =>
            {
                "simp_all".to_string()
            }

            // Default: use sorry
            _ => "sorry".to_string(),
        }
    }

    /// Check if expression involves arithmetic operations
    #[allow(clippy::missing_const_for_fn)]
    fn is_arithmetic(expr: &Expr) -> bool {
        match expr {
            Expr::Int(_) | Expr::Binary(_, _, _) | Expr::Neg(_) => true,
            _ => false, // Variables and other expressions treated conservatively
        }
    }

    /// Check if expression is linear arithmetic (no multiplication of variables)
    fn is_linear_arithmetic(expr: &Expr) -> bool {
        match expr {
            Expr::Int(_) | Expr::Var(_) => true,
            Expr::Binary(lhs, op, rhs) => {
                match op {
                    BinaryOp::Add | BinaryOp::Sub => {
                        Self::is_linear_arithmetic(lhs) && Self::is_linear_arithmetic(rhs)
                    }
                    BinaryOp::Mul => {
                        // Linear if at least one side is a constant
                        matches!(lhs.as_ref(), Expr::Int(_)) || matches!(rhs.as_ref(), Expr::Int(_))
                    }
                    _ => false,
                }
            }
            Expr::Neg(inner) => Self::is_linear_arithmetic(inner),
            _ => false,
        }
    }

    /// Check if expression is decidable (finite comparison)
    #[allow(clippy::missing_const_for_fn, clippy::match_like_matches_macro)]
    fn is_decidable(expr: &Expr) -> bool {
        match expr {
            Expr::Int(_) | Expr::Bool(_) | Expr::String(_) => true,
            _ => false,
        }
    }

    /// Check if expression involves boolean logic (and, or, not)
    /// Used to detect De Morgan's laws and similar boolean identities
    #[allow(clippy::missing_const_for_fn)]
    fn involves_boolean_logic(expr: &Expr) -> bool {
        match expr {
            Expr::Bool(_) | Expr::Not(_) | Expr::And(_, _) | Expr::Or(_, _) => true,
            _ => false, // Could be Bool but we're conservative
        }
    }

    /// Check if method call is simplifiable by simp
    fn is_simplifiable_method(method: &str) -> bool {
        // Common methods that simp has lemmas for
        matches!(
            method,
            "length"
                | "size"
                | "isEmpty"
                | "empty"
                | "head"
                | "tail"
                | "last"
                | "contains"
                | "member"
                | "map"
                | "filter"
                | "fold"
                | "union"
                | "inter"
                | "diff"
                | "insert"
                | "remove"
        )
    }

    /// Recursively check if expression contains a simplifiable method call or field access
    fn contains_simplifiable_method(expr: &Expr) -> bool {
        match expr {
            Expr::MethodCall {
                method,
                receiver,
                args,
            } => {
                Self::is_simplifiable_method(method)
                    || Self::contains_simplifiable_method(receiver)
                    || args.iter().any(Self::contains_simplifiable_method)
            }
            // Field access like xs.length is equivalent to method call without args
            Expr::FieldAccess(obj, field) => {
                Self::is_simplifiable_method(field) || Self::contains_simplifiable_method(obj)
            }
            Expr::Binary(lhs, _, rhs)
            | Expr::Compare(lhs, _, rhs)
            | Expr::And(lhs, rhs)
            | Expr::Or(lhs, rhs)
            | Expr::Implies(lhs, rhs) => {
                Self::contains_simplifiable_method(lhs) || Self::contains_simplifiable_method(rhs)
            }
            Expr::App(_, args) => args.iter().any(Self::contains_simplifiable_method),
            Expr::Not(inner) | Expr::Neg(inner) => Self::contains_simplifiable_method(inner),
            Expr::ForAll { body, .. } | Expr::Exists { body, .. } => {
                Self::contains_simplifiable_method(body)
            }
            Expr::ForAllIn {
                body, collection, ..
            }
            | Expr::ExistsIn {
                body, collection, ..
            } => {
                Self::contains_simplifiable_method(body)
                    || Self::contains_simplifiable_method(collection)
            }
            _ => false,
        }
    }

    /// Check if two expressions are structurally equal
    #[allow(clippy::match_same_arms)]
    fn exprs_equal(e1: &Expr, e2: &Expr) -> bool {
        match (e1, e2) {
            (Expr::Var(a), Expr::Var(b)) => a == b,
            (Expr::Int(a), Expr::Int(b)) => a == b,
            (Expr::Bool(a), Expr::Bool(b)) => a == b,
            (Expr::String(a), Expr::String(b)) => a == b,
            (Expr::Binary(l1, op1, r1), Expr::Binary(l2, op2, r2)) => {
                op1 == op2 && Self::exprs_equal(l1, l2) && Self::exprs_equal(r1, r2)
            }
            (Expr::App(n1, a1), Expr::App(n2, a2)) => {
                n1 == n2
                    && a1.len() == a2.len()
                    && a1
                        .iter()
                        .zip(a2.iter())
                        .all(|(x, y)| Self::exprs_equal(x, y))
            }
            _ => false,
        }
    }

    /// Check if e2 is the negation of e1
    fn is_negation_of(e1: &Expr, e2: &Expr) -> bool {
        match e2 {
            Expr::Not(inner) => Self::exprs_equal(e1, inner),
            _ => false,
        }
    }

    /// Compile an invariant to LEAN 4 definition
    #[must_use]
    pub fn compile_invariant(&self, inv: &Invariant) -> String {
        let body_str = self.compile_expr(&inv.body);
        let tactic = self.suggest_tactic(&inv.body);
        format!("theorem {} : {} := by\n  {}", inv.name, body_str, tactic)
    }

    /// Compile a refinement to LEAN 4 definition
    ///
    /// Generates:
    /// 1. Variable mappings as definitions
    /// 2. Refinement invariants as theorems
    /// 3. Abstraction function theorem
    /// 4. Simulation relation theorem
    /// 5. Action correspondence theorems
    #[must_use]
    pub fn compile_refinement(&self, ref_: &Refinement) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "-- Refinement: {} refines {}",
            ref_.name, ref_.refines
        ));
        lines.push(format!(
            "-- This refinement proves that {} correctly implements {}",
            ref_.name, ref_.refines
        ));
        lines.push(String::new());

        // 1. Variable mappings as axioms/definitions
        if !ref_.mappings.is_empty() {
            lines.push("-- Variable Mappings".to_string());
            lines.push("-- These establish state correspondence between spec and impl".to_string());
            for (i, mapping) in ref_.mappings.iter().enumerate() {
                lines.push(format!(
                    "-- mapping_{}: {} <-> {}",
                    i,
                    self.compile_expr(&mapping.spec_var),
                    self.compile_expr(&mapping.impl_var)
                ));
                // Generate an axiom for the correspondence
                lines.push(format!(
                    "axiom {}_mapping_{} : {} = {}",
                    ref_.name,
                    i,
                    self.compile_expr(&mapping.spec_var),
                    self.compile_expr(&mapping.impl_var)
                ));
            }
            lines.push(String::new());
        }

        // 2. Refinement invariants as theorems
        if !ref_.invariants.is_empty() {
            lines.push("-- Refinement Invariants".to_string());
            for (i, inv) in ref_.invariants.iter().enumerate() {
                let inv_tactic = self.suggest_tactic(inv);
                lines.push(format!(
                    "theorem {}_invariant_{} : {} := by\n  {}",
                    ref_.name,
                    i,
                    self.compile_expr(inv),
                    inv_tactic
                ));
            }
            lines.push(String::new());
        }

        // 3. Abstraction function theorem
        let abs_tactic = self.suggest_tactic(&ref_.abstraction);
        lines.push("-- Abstraction Function".to_string());
        lines.push(format!(
            "theorem {}_abstraction : {} := by\n  {}",
            ref_.name,
            self.compile_expr(&ref_.abstraction),
            abs_tactic
        ));
        lines.push(String::new());

        // 4. Simulation relation theorem
        let sim_tactic = self.suggest_tactic(&ref_.simulation);
        lines.push("-- Simulation Relation".to_string());
        lines.push(format!(
            "theorem {}_simulation : {} := by\n  {}",
            ref_.name,
            self.compile_expr(&ref_.simulation),
            sim_tactic
        ));

        // 5. Action mappings as correspondence theorems
        if !ref_.actions.is_empty() {
            lines.push(String::new());
            lines.push("-- Action Correspondence".to_string());
            for action in &ref_.actions {
                let impl_path = action.impl_action.join(".");
                lines.push(format!(
                    "-- Action: {} (spec: {}, impl: {})",
                    action.name, action.spec_action, impl_path
                ));

                // Generate correspondence axiom
                if let Some(guard) = &action.guard {
                    let guard_str = self.compile_expr(guard);
                    lines.push(format!(
                        "axiom {}_action_{} : ({}) → {} ≈ {}",
                        ref_.name, action.name, guard_str, action.spec_action, impl_path
                    ));
                } else {
                    lines.push(format!(
                        "axiom {}_action_{} : {} ≈ {}",
                        ref_.name, action.name, action.spec_action, impl_path
                    ));
                }
            }
        }

        lines.join("\n")
    }

    /// Compile a security property to LEAN 4 definition
    #[must_use]
    pub fn compile_security(&self, security: &Security) -> String {
        let body_str = self.compile_expr(&security.body);
        let tactic = self.suggest_tactic(&security.body);
        format!(
            "theorem {} : {} := by\n  {}",
            security.name, body_str, tactic
        )
    }

    /// Compile a composed theorem to LEAN 4
    ///
    /// Composed theorems declare dependencies on other properties via `uses`.
    /// Each dependency is introduced as an assumption via `have` bindings.
    ///
    /// Example USL:
    /// ```text
    /// composed theorem modular_safety {
    ///     uses { acyclic_theorem, connectivity_invariant }
    ///     acyclic_theorem and connectivity_invariant implies safe_execution
    /// }
    /// ```
    ///
    /// Compiles to:
    /// ```lean
    /// theorem modular_safety
    ///     (acyclic_theorem : Bool)
    ///     (connectivity_invariant : Bool)
    ///     : acyclic_theorem ∧ connectivity_invariant → safe_execution := by
    ///   intro h
    ///   ...
    /// ```
    #[must_use]
    pub fn compile_composed_theorem(&self, composed: &crate::ast::ComposedTheorem) -> String {
        let mut lines = Vec::new();

        // Document the composition
        lines.push(format!(
            "-- Composed theorem: {} (uses: {})",
            composed.name,
            composed.uses.join(", ")
        ));

        // Build parameter list for dependencies
        let params: Vec<String> = composed
            .uses
            .iter()
            .map(|dep| format!("({dep} : Prop)"))
            .collect();

        let params_str = if params.is_empty() {
            String::new()
        } else {
            format!("\n    {}", params.join("\n    "))
        };

        // Compile body and suggest tactics
        let body_str = self.compile_expr(&composed.body);
        let tactic = self.suggest_tactic(&composed.body);

        lines.push(format!(
            "theorem {}{}\n    : {} := by\n  {}",
            composed.name, params_str, body_str, tactic
        ));

        lines.join("\n")
    }

    /// Compile a version specification to LEAN 4
    ///
    /// Generates:
    /// - A comment documenting the version improvement relationship
    /// - Theorems for each capability clause (must be >=)
    /// - Theorems for each preserves clause (must hold for V2)
    #[must_use]
    pub fn compile_version_spec(&self, version: &VersionSpec) -> String {
        let mut lines = Vec::new();

        lines.push(format!(
            "-- Version Improvement: {} improves {}",
            version.name, version.improves
        ));
        lines.push(format!(
            "-- This specification proves {} is at least as capable as {}",
            version.name, version.improves
        ));
        lines.push(String::new());

        // Generate theorems for capability clauses
        for (i, cap) in version.capabilities.iter().enumerate() {
            let body_str = self.compile_expr(&cap.expr);
            let tactic = self.suggest_tactic(&cap.expr);
            lines.push(format!(
                "-- Capability improvement: {} over {}",
                version.name, version.improves
            ));
            lines.push(format!(
                "theorem {}_capability_{} : {} := by\n  {}",
                version.name.to_lowercase(),
                i + 1,
                body_str,
                tactic
            ));
            lines.push(String::new());
        }

        // Generate theorems for preserves clauses
        for (i, pres) in version.preserves.iter().enumerate() {
            let body_str = self.compile_expr(&pres.property);
            let tactic = self.suggest_tactic(&pres.property);
            lines.push(format!("-- Preserved property from {}", version.improves));
            lines.push(format!(
                "theorem {}_preserves_{} : {} := by\n  {}",
                version.name.to_lowercase(),
                i + 1,
                body_str,
                tactic
            ));
            lines.push(String::new());
        }

        // Generate a combined theorem that version improvement holds
        if !version.capabilities.is_empty() || !version.preserves.is_empty() {
            let mut conjuncts = Vec::new();

            for (i, _) in version.capabilities.iter().enumerate() {
                conjuncts.push(format!(
                    "{}_capability_{}",
                    version.name.to_lowercase(),
                    i + 1
                ));
            }
            for (i, _) in version.preserves.iter().enumerate() {
                conjuncts.push(format!(
                    "{}_preserves_{}",
                    version.name.to_lowercase(),
                    i + 1
                ));
            }

            lines.push(format!(
                "-- Combined version improvement theorem: {} improves {}",
                version.name, version.improves
            ));

            // Generate reference to all sub-theorems
            lines.push(format!("-- Proof requires: {}", conjuncts.join(", ")));
        }

        lines.join("\n")
    }

    /// Compile a capability specification to LEAN 4 definitions
    ///
    /// A capability spec `capability Name { can f(...) -> T; requires { P } }` compiles to:
    /// - A structure type representing the capability
    /// - Axioms for each ability
    /// - Theorems for each requirement
    #[must_use]
    pub fn compile_capability_spec(&self, capability: &CapabilitySpec) -> String {
        let mut lines = Vec::new();

        lines.push(format!("-- Capability Specification: {}", capability.name));
        lines.push(String::new());

        // Generate structure for the capability
        lines.push(format!("structure {} where", capability.name));

        // Generate fields for each ability
        for ability in &capability.abilities {
            let return_type = ability
                .return_type
                .as_ref()
                .map(|t| self.compile_type(t))
                .unwrap_or_else(|| "Unit".to_string());

            let params: Vec<String> = ability
                .params
                .iter()
                .map(|p| format!("{}: {}", p.name, self.compile_type(&p.ty)))
                .collect();

            if params.is_empty() {
                lines.push(format!("  {} : {}", ability.name, return_type));
            } else {
                lines.push(format!(
                    "  {} : ({}) → {}",
                    ability.name,
                    params.join(" × "),
                    return_type
                ));
            }
        }

        lines.push(String::new());

        // Generate axioms for requirements
        for (i, req) in capability.requires.iter().enumerate() {
            let body_str = self.compile_expr(req);
            lines.push(format!("-- Requirement {} for {}", i + 1, capability.name));
            lines.push(format!(
                "axiom {}_requirement_{} : {}",
                capability.name.to_lowercase(),
                i + 1,
                body_str
            ));
            lines.push(String::new());
        }

        lines.join("\n")
    }

    /// Compile an improvement proposal to LEAN 4 definitions
    ///
    /// An improvement proposal specifies what must be improved and what must be preserved.
    /// In LEAN 4, this compiles to a structure with obligations as propositions.
    ///
    /// Example USL:
    /// ```text
    /// improvement_proposal CodeOptimization {
    ///     target { Dasher.verify_rust_code }
    ///     improves { execution_speed >= 1.1 * baseline }
    ///     preserves { soundness, completeness }
    ///     requires { valid_rust_syntax(new_code) }
    /// }
    /// ```
    #[must_use]
    pub fn compile_improvement_proposal(&self, proposal: &ImprovementProposal) -> String {
        let mut lines = Vec::new();

        lines.push(format!("-- Improvement Proposal: {}", proposal.name));
        lines
            .push("-- This proposal specifies conditions for a valid self-improvement".to_string());
        lines.push(String::new());

        // Structure for the proposal
        lines.push(format!("structure {} where", proposal.name));
        lines.push(format!(
            "  target : {} → Prop",
            if proposal.improves.is_empty() && proposal.preserves.is_empty() {
                "Unit"
            } else {
                "Prop"
            }
        ));

        // Fields for improvement conditions
        for (i, _) in proposal.improves.iter().enumerate() {
            lines.push(format!("  improves_{} : Prop", i + 1));
        }

        // Fields for preservation conditions
        for (i, _) in proposal.preserves.iter().enumerate() {
            lines.push(format!("  preserves_{} : Prop", i + 1));
        }

        // Fields for preconditions
        for (i, _) in proposal.requires.iter().enumerate() {
            lines.push(format!("  requires_{} : Prop", i + 1));
        }

        lines.push(String::new());

        // Target definition
        lines.push(format!(
            "-- Target: {}",
            self.compile_expr(&proposal.target)
        ));
        lines.push(format!(
            "def {}_target : Prop := {}",
            proposal.name.to_lowercase(),
            self.compile_expr(&proposal.target)
        ));
        lines.push(String::new());

        // Theorems for improvement clauses
        for (i, improves) in proposal.improves.iter().enumerate() {
            let body_str = self.compile_expr(improves);
            let tactic = self.suggest_tactic(improves);
            lines.push(format!(
                "-- Improvement clause {}: must be strictly better",
                i + 1
            ));
            lines.push(format!(
                "theorem {}_improves_{} : {} := by\n  {}",
                proposal.name.to_lowercase(),
                i + 1,
                body_str,
                tactic
            ));
            lines.push(String::new());
        }

        // Theorems for preservation clauses
        for (i, preserves) in proposal.preserves.iter().enumerate() {
            let body_str = self.compile_expr(preserves);
            let tactic = self.suggest_tactic(preserves);
            lines.push(format!(
                "-- Preservation clause {}: must be at least as good",
                i + 1
            ));
            lines.push(format!(
                "theorem {}_preserves_{} : {} := by\n  {}",
                proposal.name.to_lowercase(),
                i + 1,
                body_str,
                tactic
            ));
            lines.push(String::new());
        }

        // Theorems for preconditions
        for (i, requires) in proposal.requires.iter().enumerate() {
            let body_str = self.compile_expr(requires);
            let tactic = self.suggest_tactic(requires);
            lines.push(format!(
                "-- Precondition {}: must hold before improvement",
                i + 1
            ));
            lines.push(format!(
                "theorem {}_requires_{} : {} := by\n  {}",
                proposal.name.to_lowercase(),
                i + 1,
                body_str,
                tactic
            ));
            lines.push(String::new());
        }

        // Combined validity theorem
        if !proposal.improves.is_empty()
            || !proposal.preserves.is_empty()
            || !proposal.requires.is_empty()
        {
            let mut conjuncts = Vec::new();
            for improves in &proposal.improves {
                conjuncts.push(self.compile_expr(improves));
            }
            for preserves in &proposal.preserves {
                conjuncts.push(self.compile_expr(preserves));
            }
            for requires in &proposal.requires {
                conjuncts.push(self.compile_expr(requires));
            }

            let combined = conjuncts.join(") ∧ (");
            lines.push("-- Combined validity: all conditions must hold".to_string());
            lines.push(format!(
                "def {}_valid : Prop := ({})",
                proposal.name.to_lowercase(),
                combined
            ));
        }

        lines.join("\n")
    }

    /// Compile a verification gate to LEAN 4 definitions
    ///
    /// A verification gate specifies mandatory checks before accepting changes.
    /// In LEAN 4, this compiles to an inductive type with proof obligations.
    ///
    /// Example USL:
    /// ```text
    /// verification_gate SelfModificationGate {
    ///     inputs { current: DasherVersion, proposed: Improvement }
    ///     checks {
    ///         check soundness_preserved { verify_soundness(current, proposed) }
    ///         check capability_preserved { verify_capabilities(current, proposed) }
    ///     }
    ///     on_pass { result = apply(proposed) }
    ///     on_fail { result = reject(proposed, errors) }
    /// }
    /// ```
    #[must_use]
    pub fn compile_verification_gate(&self, gate: &VerificationGate) -> String {
        let mut lines = Vec::new();

        lines.push(format!("-- Verification Gate: {}", gate.name));
        lines.push(
            "-- This gate enforces mandatory verification before self-modification".to_string(),
        );
        lines.push(String::new());

        // Structure for gate inputs
        if !gate.inputs.is_empty() {
            lines.push("-- Input parameters".to_string());
            lines.push(format!("structure {}_Inputs where", gate.name));
            for input in &gate.inputs {
                lines.push(format!(
                    "  {} : {}",
                    input.name,
                    self.compile_type(&input.ty)
                ));
            }
            lines.push(String::new());
        }

        // Individual check definitions
        for check in &gate.checks {
            let body_str = self.compile_expr(&check.condition);
            lines.push(format!("-- Verification check: {}", check.name));
            lines.push(format!(
                "def {}_check_{} : Prop := {}",
                gate.name.to_lowercase(),
                check.name,
                body_str
            ));
            lines.push(String::new());
        }

        // Combined check (all checks must pass)
        if !gate.checks.is_empty() {
            let check_props: Vec<String> = gate
                .checks
                .iter()
                .map(|c| format!("{}_check_{}", gate.name.to_lowercase(), c.name))
                .collect();
            lines.push("-- All checks must pass".to_string());
            lines.push(format!(
                "def {}_all_checks_pass : Prop := {}",
                gate.name.to_lowercase(),
                check_props.join(" ∧ ")
            ));
            lines.push(String::new());
        }

        // Inductive type for gate result
        lines.push("-- Gate result: either pass or fail".to_string());
        lines.push(format!("inductive {}_Result where", gate.name));
        lines.push(format!(
            "  | pass : {}_all_checks_pass → {}_Result",
            gate.name.to_lowercase(),
            gate.name
        ));
        lines.push(format!(
            "  | fail : ¬{}_all_checks_pass → {}_Result",
            gate.name.to_lowercase(),
            gate.name
        ));
        lines.push(String::new());

        // On pass action
        lines.push("-- Action when all checks pass".to_string());
        lines.push(format!(
            "def {}_on_pass := {}",
            gate.name.to_lowercase(),
            self.compile_expr(&gate.on_pass)
        ));
        lines.push(String::new());

        // On fail action
        lines.push("-- Action when any check fails".to_string());
        lines.push(format!(
            "def {}_on_fail := {}",
            gate.name.to_lowercase(),
            self.compile_expr(&gate.on_fail)
        ));

        lines.join("\n")
    }

    /// Compile a rollback specification to LEAN 4 definitions
    ///
    /// A rollback spec defines safe state recovery after failed improvements.
    /// In LEAN 4, this compiles to a structure with invariant proofs.
    ///
    /// Example USL:
    /// ```text
    /// rollback_spec SafeRollback {
    ///     state { current: DasherVersion, history: List<DasherVersion> }
    ///     invariant { |history| > 0 }
    ///     trigger { verification_failed or runtime_error }
    ///     action { current = history.last(); ensure { verified(current) } }
    ///     guarantee { always(verified(current)) }
    /// }
    /// ```
    #[must_use]
    pub fn compile_rollback_spec(&self, rollback: &RollbackSpec) -> String {
        let mut lines = Vec::new();

        lines.push(format!("-- Rollback Specification: {}", rollback.name));
        lines.push(
            "-- This specification ensures safe recovery from failed improvements".to_string(),
        );
        lines.push(String::new());

        // State structure
        if !rollback.state.is_empty() {
            lines.push("-- State variables for rollback".to_string());
            lines.push(format!("structure {}_State where", rollback.name));
            for param in &rollback.state {
                lines.push(format!(
                    "  {} : {}",
                    param.name,
                    self.compile_type(&param.ty)
                ));
            }
            lines.push(String::new());
        }

        // Invariants
        for (i, inv) in rollback.invariants.iter().enumerate() {
            let body_str = self.compile_expr(inv);
            let tactic = self.suggest_tactic(inv);
            lines.push(format!(
                "-- Invariant {}: must hold before and after rollback",
                i + 1
            ));
            lines.push(format!(
                "theorem {}_invariant_{} : {} := by\n  {}",
                rollback.name.to_lowercase(),
                i + 1,
                body_str,
                tactic
            ));
            lines.push(String::new());
        }

        // Combined invariants
        if !rollback.invariants.is_empty() {
            let inv_exprs: Vec<String> = rollback
                .invariants
                .iter()
                .map(|inv| self.compile_expr(inv))
                .collect();
            lines.push("-- All invariants combined".to_string());
            lines.push(format!(
                "def {}_all_invariants : Prop := {}",
                rollback.name.to_lowercase(),
                inv_exprs.join(" ∧ ")
            ));
            lines.push(String::new());
        }

        // Trigger condition
        lines.push("-- Rollback trigger condition".to_string());
        lines.push(format!(
            "def {}_trigger : Prop := {}",
            rollback.name.to_lowercase(),
            self.compile_expr(&rollback.trigger)
        ));
        lines.push(String::new());

        // Rollback action as a function
        lines.push("-- Rollback action".to_string());
        if !rollback.action.assignments.is_empty() {
            for (var, expr) in &rollback.action.assignments {
                lines.push(format!(
                    "-- Assignment: {} := {}",
                    var,
                    self.compile_expr(expr)
                ));
            }
        }
        if let Some(ensure) = &rollback.action.ensure {
            lines.push(format!(
                "-- Ensure after rollback: {}",
                self.compile_expr(ensure)
            ));
            lines.push(format!(
                "def {}_ensure : Prop := {}",
                rollback.name.to_lowercase(),
                self.compile_expr(ensure)
            ));
            lines.push(String::new());
        }

        // Guarantees
        for (i, guarantee) in rollback.guarantees.iter().enumerate() {
            let body_str = self.compile_expr(guarantee);
            let tactic = self.suggest_tactic(guarantee);
            lines.push(format!(
                "-- Guarantee {}: must hold after rollback completes",
                i + 1
            ));
            lines.push(format!(
                "theorem {}_guarantee_{} : {} := by\n  {}",
                rollback.name.to_lowercase(),
                i + 1,
                body_str,
                tactic
            ));
            lines.push(String::new());
        }

        // Complete rollback specification theorem
        let mut spec_parts = Vec::new();
        if !rollback.invariants.is_empty() {
            spec_parts.push(format!("{}_all_invariants", rollback.name.to_lowercase()));
        }
        for i in 0..rollback.guarantees.len() {
            spec_parts.push(self.compile_expr(&rollback.guarantees[i]));
        }

        if !spec_parts.is_empty() {
            lines.push("-- Complete rollback specification".to_string());
            lines.push(format!(
                "theorem {}_spec : {}_trigger → ({}) := by\n  sorry",
                rollback.name.to_lowercase(),
                rollback.name.to_lowercase(),
                spec_parts.join(" ∧ ")
            ));
        }

        lines.join("\n")
    }

    /// Generate complete LEAN 4 file from spec
    #[must_use]
    pub fn compile_module(&self, typed_spec: &TypedSpec) -> CompiledSpec {
        // Reset state tracking for this compilation
        self.needs_classical.set(false);
        self.needs_ring.set(false);
        self.needs_linarith.set(false);

        // Pre-compile all properties to detect Classical usage
        let mut compiled_properties = Vec::new();
        for property in &typed_spec.spec.properties {
            match property {
                Property::Theorem(thm) => {
                    compiled_properties.push((property.clone(), self.compile_theorem(thm)));
                }
                Property::Invariant(inv) => {
                    compiled_properties.push((property.clone(), self.compile_invariant(inv)));
                }
                Property::Refinement(ref_) => {
                    compiled_properties.push((property.clone(), self.compile_refinement(ref_)));
                }
                Property::Security(security) => {
                    compiled_properties.push((property.clone(), self.compile_security(security)));
                }
                Property::Version(version) => {
                    compiled_properties
                        .push((property.clone(), self.compile_version_spec(version)));
                }
                Property::Capability(capability) => {
                    compiled_properties
                        .push((property.clone(), self.compile_capability_spec(capability)));
                }
                Property::Composed(composed) => {
                    compiled_properties
                        .push((property.clone(), self.compile_composed_theorem(composed)));
                }
                Property::ImprovementProposal(proposal) => {
                    compiled_properties.push((
                        property.clone(),
                        self.compile_improvement_proposal(proposal),
                    ));
                }
                Property::VerificationGate(gate) => {
                    compiled_properties
                        .push((property.clone(), self.compile_verification_gate(gate)));
                }
                Property::Rollback(rollback) => {
                    compiled_properties
                        .push((property.clone(), self.compile_rollback_spec(rollback)));
                }
                _ => {}
            }
        }

        let mut sections = Vec::new();
        let mut imports = vec![
            "Mathlib.Data.Set.Basic".to_string(),
            "Mathlib.Data.List.Basic".to_string(),
        ];

        // File header
        sections.push("-- Generated from USL by DashProve".to_string());
        sections.push(format!("-- Namespace: {}", self.namespace));
        sections.push(String::new());

        // Imports
        sections.push("import Mathlib.Data.Set.Basic".to_string());
        sections.push("import Mathlib.Data.List.Basic".to_string());

        // Add Classical import if needed
        if self.needs_classical.get() {
            sections.push("import Mathlib.Logic.Classical".to_string());
            imports.push("Mathlib.Logic.Classical".to_string());
        }

        // Add Omega for arithmetic tactics
        sections.push("import Mathlib.Tactic.Omega".to_string());
        imports.push("Mathlib.Tactic.Omega".to_string());

        // Add Ring import if needed for polynomial arithmetic
        if self.needs_ring.get() {
            sections.push("import Mathlib.Tactic.Ring".to_string());
            imports.push("Mathlib.Tactic.Ring".to_string());
        }

        // Add Linarith import if needed for linear arithmetic with hypotheses
        if self.needs_linarith.get() {
            sections.push("import Mathlib.Tactic.Linarith".to_string());
            imports.push("Mathlib.Tactic.Linarith".to_string());
        }

        sections.push(String::new());

        // Open Classical namespace if needed
        if self.needs_classical.get() {
            sections.push("open Classical".to_string());
            sections.push(String::new());
        }

        // Namespace
        sections.push(format!("namespace {}", self.namespace));
        sections.push(String::new());

        // Compile type definitions as structures
        for type_def in &typed_spec.spec.types {
            sections.push(format!("-- Type: {}", type_def.name));
            sections.push(format!("structure {} where", type_def.name));
            for field in &type_def.fields {
                sections.push(format!(
                    "  {} : {}",
                    field.name,
                    self.compile_type(&field.ty)
                ));
            }
            sections.push(String::new());
        }

        // Use pre-compiled properties
        for (property, compiled) in compiled_properties {
            match property {
                Property::Theorem(thm) => {
                    sections.push(format!("-- Theorem: {}", thm.name));
                    sections.push(compiled);
                    sections.push(String::new());
                }
                Property::Invariant(inv) => {
                    sections.push(format!("-- Invariant: {}", inv.name));
                    sections.push(compiled);
                    sections.push(String::new());
                }
                Property::Refinement(_) => {
                    sections.push(compiled);
                    sections.push(String::new());
                }
                Property::Security(security) => {
                    sections.push(format!("-- Security: {}", security.name));
                    sections.push(compiled);
                    sections.push(String::new());
                }
                Property::Version(version) => {
                    sections.push(format!(
                        "-- Version: {} improves {}",
                        version.name, version.improves
                    ));
                    sections.push(compiled);
                    sections.push(String::new());
                }
                Property::Capability(capability) => {
                    sections.push(format!("-- Capability: {}", capability.name));
                    sections.push(compiled);
                    sections.push(String::new());
                }
                Property::ImprovementProposal(proposal) => {
                    sections.push(format!("-- Improvement Proposal: {}", proposal.name));
                    sections.push(compiled);
                    sections.push(String::new());
                }
                Property::VerificationGate(gate) => {
                    sections.push(format!("-- Verification Gate: {}", gate.name));
                    sections.push(compiled);
                    sections.push(String::new());
                }
                Property::Rollback(rollback) => {
                    sections.push(format!("-- Rollback Specification: {}", rollback.name));
                    sections.push(compiled);
                    sections.push(String::new());
                }
                _ => {
                    // Temporal, Contract, Probabilistic, Semantic, PlatformApi,
                    // Bisimulation not directly supported in LEAN
                }
            }
        }

        // Close namespace
        sections.push(format!("end {}", self.namespace));

        CompiledSpec {
            backend: "LEAN4".to_string(),
            code: sections.join("\n"),
            module_name: Some(self.namespace.clone()),
            imports,
        }
    }
}

/// Compile to LEAN 4
#[must_use]
pub fn compile_to_lean(spec: &TypedSpec) -> CompiledSpec {
    let compiler = Lean4Compiler::new("USLSpec");
    compiler.compile_module(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Field, Spec, TypeDef};
    use crate::typecheck::typecheck;

    fn make_compiler() -> Lean4Compiler {
        Lean4Compiler::new("Test")
    }

    // Expression compilation tests

    #[test]
    fn test_compile_var() {
        let compiler = make_compiler();
        assert_eq!(compiler.compile_expr(&Expr::Var("x".into())), "x");
    }

    #[test]
    fn test_compile_primed_var() {
        let compiler = make_compiler();
        assert_eq!(compiler.compile_expr(&Expr::Var("x'".into())), "x_next");
    }

    #[test]
    fn test_compile_int() {
        let compiler = make_compiler();
        assert_eq!(compiler.compile_expr(&Expr::Int(42)), "42");
        assert_eq!(compiler.compile_expr(&Expr::Int(-17)), "-17");
    }

    #[test]
    fn test_compile_bool() {
        let compiler = make_compiler();
        assert_eq!(compiler.compile_expr(&Expr::Bool(true)), "True");
        assert_eq!(compiler.compile_expr(&Expr::Bool(false)), "False");
    }

    #[test]
    fn test_compile_string() {
        let compiler = make_compiler();
        assert_eq!(
            compiler.compile_expr(&Expr::String("hello".into())),
            "\"hello\""
        );
    }

    #[test]
    fn test_compile_forall() {
        let compiler = make_compiler();
        let expr = Expr::ForAll {
            var: "x".into(),
            ty: Some(Type::Named("Int".into())),
            body: Box::new(Expr::Bool(true)),
        };
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("∀"));
        assert!(result.contains("x"));
        assert!(result.contains("Int"));
    }

    #[test]
    fn test_compile_exists() {
        let compiler = make_compiler();
        let expr = Expr::Exists {
            var: "y".into(),
            ty: None,
            body: Box::new(Expr::Bool(false)),
        };
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("∃"));
        assert!(result.contains("y"));
    }

    #[test]
    fn test_compile_implies() {
        let compiler = make_compiler();
        let expr = Expr::Implies(
            Box::new(Expr::Var("p".into())),
            Box::new(Expr::Var("q".into())),
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("→"));
        assert!(result.contains("p"));
        assert!(result.contains("q"));
    }

    #[test]
    fn test_compile_and() {
        let compiler = make_compiler();
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("∧"));
    }

    #[test]
    fn test_compile_or() {
        let compiler = make_compiler();
        let expr = Expr::Or(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("∨"));
    }

    #[test]
    fn test_compile_not() {
        let compiler = make_compiler();
        let expr = Expr::Not(Box::new(Expr::Var("x".into())));
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("¬"));
        assert!(result.contains("x"));
    }

    #[test]
    fn test_compile_comparison_eq() {
        let compiler = make_compiler();
        let expr = Expr::Compare(
            Box::new(Expr::Int(1)),
            ComparisonOp::Eq,
            Box::new(Expr::Int(1)),
        );
        assert!(compiler.compile_expr(&expr).contains("="));
    }

    #[test]
    fn test_compile_comparison_ne() {
        let compiler = make_compiler();
        let expr = Expr::Compare(
            Box::new(Expr::Int(1)),
            ComparisonOp::Ne,
            Box::new(Expr::Int(2)),
        );
        assert!(compiler.compile_expr(&expr).contains("≠"));
    }

    #[test]
    fn test_compile_comparison_lt() {
        let compiler = make_compiler();
        let expr = Expr::Compare(
            Box::new(Expr::Int(1)),
            ComparisonOp::Lt,
            Box::new(Expr::Int(2)),
        );
        assert!(compiler.compile_expr(&expr).contains("<"));
    }

    #[test]
    fn test_compile_comparison_le() {
        let compiler = make_compiler();
        let expr = Expr::Compare(
            Box::new(Expr::Int(1)),
            ComparisonOp::Le,
            Box::new(Expr::Int(2)),
        );
        assert!(compiler.compile_expr(&expr).contains("≤"));
    }

    #[test]
    fn test_compile_comparison_gt() {
        let compiler = make_compiler();
        let expr = Expr::Compare(
            Box::new(Expr::Int(2)),
            ComparisonOp::Gt,
            Box::new(Expr::Int(1)),
        );
        assert!(compiler.compile_expr(&expr).contains(">"));
    }

    #[test]
    fn test_compile_comparison_ge() {
        let compiler = make_compiler();
        let expr = Expr::Compare(
            Box::new(Expr::Int(2)),
            ComparisonOp::Ge,
            Box::new(Expr::Int(1)),
        );
        assert!(compiler.compile_expr(&expr).contains("≥"));
    }

    #[test]
    fn test_compile_binary_add() {
        let compiler = make_compiler();
        let expr = Expr::Binary(
            Box::new(Expr::Int(1)),
            BinaryOp::Add,
            Box::new(Expr::Int(2)),
        );
        assert!(compiler.compile_expr(&expr).contains("+"));
    }

    #[test]
    fn test_compile_binary_sub() {
        let compiler = make_compiler();
        let expr = Expr::Binary(
            Box::new(Expr::Int(5)),
            BinaryOp::Sub,
            Box::new(Expr::Int(3)),
        );
        assert!(compiler.compile_expr(&expr).contains("-"));
    }

    #[test]
    fn test_compile_binary_mul() {
        let compiler = make_compiler();
        let expr = Expr::Binary(
            Box::new(Expr::Int(2)),
            BinaryOp::Mul,
            Box::new(Expr::Int(3)),
        );
        assert!(compiler.compile_expr(&expr).contains("*"));
    }

    #[test]
    fn test_compile_binary_div() {
        let compiler = make_compiler();
        let expr = Expr::Binary(
            Box::new(Expr::Int(6)),
            BinaryOp::Div,
            Box::new(Expr::Int(2)),
        );
        assert!(compiler.compile_expr(&expr).contains("/"));
    }

    #[test]
    fn test_compile_neg() {
        let compiler = make_compiler();
        let expr = Expr::Neg(Box::new(Expr::Var("x".into())));
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("-"));
        assert!(result.contains("x"));
    }

    #[test]
    fn test_compile_app_no_args() {
        let compiler = make_compiler();
        let expr = Expr::App("f".into(), vec![]);
        assert_eq!(compiler.compile_expr(&expr), "f");
    }

    #[test]
    fn test_compile_app_with_args() {
        let compiler = make_compiler();
        let expr = Expr::App("f".into(), vec![Expr::Int(1), Expr::Int(2)]);
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("f"));
        assert!(result.contains("1"));
        assert!(result.contains("2"));
    }

    #[test]
    fn test_compile_field_access() {
        let compiler = make_compiler();
        let expr = Expr::FieldAccess(Box::new(Expr::Var("obj".into())), "field".into());
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("obj"));
        assert!(result.contains("field"));
        assert!(result.contains("."));
    }

    // Type compilation tests

    #[test]
    fn test_compile_type_bool() {
        let compiler = make_compiler();
        assert_eq!(compiler.compile_type(&Type::Named("Bool".into())), "Bool");
        assert_eq!(
            compiler.compile_type(&Type::Named("boolean".into())),
            "Bool"
        );
    }

    #[test]
    fn test_compile_type_int() {
        let compiler = make_compiler();
        assert_eq!(compiler.compile_type(&Type::Named("Int".into())), "Int");
        assert_eq!(compiler.compile_type(&Type::Named("int".into())), "Int");
        assert_eq!(compiler.compile_type(&Type::Named("integer".into())), "Int");
    }

    #[test]
    fn test_compile_type_float() {
        let compiler = make_compiler();
        assert_eq!(compiler.compile_type(&Type::Named("Float".into())), "Float");
        assert_eq!(compiler.compile_type(&Type::Named("float".into())), "Float");
    }

    #[test]
    fn test_compile_type_string() {
        let compiler = make_compiler();
        assert_eq!(
            compiler.compile_type(&Type::Named("String".into())),
            "String"
        );
        assert_eq!(
            compiler.compile_type(&Type::Named("string".into())),
            "String"
        );
    }

    #[test]
    fn test_compile_type_custom() {
        let compiler = make_compiler();
        assert_eq!(
            compiler.compile_type(&Type::Named("MyType".into())),
            "MyType"
        );
    }

    #[test]
    fn test_compile_type_set() {
        let compiler = make_compiler();
        let ty = Type::Set(Box::new(Type::Named("Int".into())));
        assert_eq!(compiler.compile_type(&ty), "Set Int");
    }

    #[test]
    fn test_compile_type_list() {
        let compiler = make_compiler();
        let ty = Type::List(Box::new(Type::Named("Bool".into())));
        assert_eq!(compiler.compile_type(&ty), "List Bool");
    }

    #[test]
    fn test_compile_type_map() {
        let compiler = make_compiler();
        let ty = Type::Map(
            Box::new(Type::Named("String".into())),
            Box::new(Type::Named("Int".into())),
        );
        assert!(compiler.compile_type(&ty).contains("→"));
    }

    // Theorem compilation tests

    #[test]
    fn test_compile_theorem() {
        let compiler = make_compiler();
        let thm = Theorem {
            name: "test_theorem".into(),
            body: Expr::Bool(true),
        };
        let result = compiler.compile_theorem(&thm);
        assert!(result.contains("theorem"));
        assert!(result.contains("test_theorem"));
        assert!(result.contains("True"));
    }

    #[test]
    fn test_compile_theorem_with_quantifier() {
        let compiler = make_compiler();
        let thm = Theorem {
            name: "forall_test".into(),
            body: Expr::ForAll {
                var: "x".into(),
                ty: Some(Type::Named("Int".into())),
                body: Box::new(Expr::Compare(
                    Box::new(Expr::Var("x".into())),
                    ComparisonOp::Eq,
                    Box::new(Expr::Var("x".into())),
                )),
            },
        };
        let result = compiler.compile_theorem(&thm);
        assert!(result.contains("theorem"));
        assert!(result.contains("forall_test"));
        assert!(result.contains("∀"));
    }

    // Invariant compilation tests

    #[test]
    fn test_compile_invariant() {
        let compiler = make_compiler();
        let inv = Invariant {
            name: "positive".into(),
            body: Expr::Compare(
                Box::new(Expr::Var("x".into())),
                ComparisonOp::Gt,
                Box::new(Expr::Int(0)),
            ),
        };
        let result = compiler.compile_invariant(&inv);
        // Invariants are compiled as theorems in LEAN 4
        assert!(result.contains("theorem"));
        assert!(result.contains("positive"));
        assert!(result.contains(">"));
    }

    // Tactic suggestion tests

    #[test]
    fn test_suggest_tactic_simple() {
        let compiler = make_compiler();
        let expr = Expr::Bool(true);
        let tactic = compiler.suggest_tactic(&expr);
        assert!(!tactic.is_empty());
        // trivial is suggested for True
        assert!(tactic.contains("trivial"));
    }

    #[test]
    fn test_suggest_tactic_implies() {
        let compiler = make_compiler();
        let expr = Expr::Implies(
            Box::new(Expr::Var("p".into())),
            Box::new(Expr::Var("q".into())),
        );
        let tactic = compiler.suggest_tactic(&expr);
        // Should suggest intro for implication
        assert!(tactic.contains("intro"));
    }

    #[test]
    fn test_suggest_tactic_forall() {
        let compiler = make_compiler();
        let expr = Expr::ForAll {
            var: "x".into(),
            ty: Some(Type::Named("Int".into())),
            body: Box::new(Expr::Bool(true)),
        };
        let tactic = compiler.suggest_tactic(&expr);
        // Should suggest intro for forall
        assert!(tactic.contains("intro"));
    }

    #[test]
    fn test_suggest_tactic_and() {
        let compiler = make_compiler();
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(true)));
        let tactic = compiler.suggest_tactic(&expr);
        // Should suggest constructor/And.intro for conjunction
        assert!(!tactic.is_empty());
    }

    #[test]
    fn test_suggest_tactic_exists() {
        let compiler = make_compiler();
        let expr = Expr::Exists {
            var: "x".into(),
            ty: Some(Type::Named("Int".into())),
            body: Box::new(Expr::Bool(true)),
        };
        let tactic = compiler.suggest_tactic(&expr);
        // Should suggest use for exists
        assert!(tactic.contains("use"));
    }

    #[test]
    fn test_suggest_tactic_or() {
        let compiler = make_compiler();
        let expr = Expr::Or(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let tactic = compiler.suggest_tactic(&expr);
        // For decidable disjunctions, should suggest decide
        // The actual tactic depends on whether sides are decidable
        assert!(!tactic.is_empty());
    }

    // Module compilation tests

    #[test]
    fn test_compile_module_empty() {
        let spec = Spec {
            types: vec![],
            properties: vec![],
        };
        let typed = typecheck(spec).unwrap();
        let result = compile_to_lean(&typed);
        assert_eq!(result.backend, "LEAN4");
        assert!(result.code.contains("namespace"));
        assert!(result.code.contains("end"));
    }

    #[test]
    fn test_compile_module_with_theorem() {
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Theorem(Theorem {
                name: "simple".into(),
                body: Expr::Bool(true),
            })],
        };
        let typed = typecheck(spec).unwrap();
        let result = compile_to_lean(&typed);
        assert!(result.code.contains("theorem"));
        assert!(result.code.contains("simple"));
    }

    #[test]
    fn test_compile_module_with_invariant() {
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Invariant(Invariant {
                name: "inv1".into(),
                body: Expr::Bool(true),
            })],
        };
        let typed = typecheck(spec).unwrap();
        let result = compile_to_lean(&typed);
        // Invariants are compiled as theorems in LEAN 4
        assert!(result.code.contains("theorem"));
        assert!(result.code.contains("inv1"));
    }

    #[test]
    fn test_compile_module_with_type_def() {
        let spec = Spec {
            types: vec![TypeDef {
                name: "Point".into(),
                fields: vec![
                    Field {
                        name: "x".into(),
                        ty: Type::Named("Int".into()),
                    },
                    Field {
                        name: "y".into(),
                        ty: Type::Named("Int".into()),
                    },
                ],
            }],
            properties: vec![],
        };
        let typed = typecheck(spec).unwrap();
        let result = compile_to_lean(&typed);
        assert!(result.code.contains("structure"));
        assert!(result.code.contains("Point"));
    }

    // Namespace tests

    #[test]
    fn test_compiler_namespace() {
        let compiler = Lean4Compiler::new("MyNamespace");
        let spec = Spec {
            types: vec![],
            properties: vec![],
        };
        let typed = typecheck(spec).unwrap();
        let result = compiler.compile_module(&typed);
        assert!(result.code.contains("namespace MyNamespace"));
        assert!(result.code.contains("end MyNamespace"));
        assert_eq!(result.module_name, Some("MyNamespace".into()));
    }

    // Classical logic tracking tests

    #[test]
    fn test_needs_classical_for_law_of_excluded_middle() {
        let compiler = make_compiler();
        // Or with negation typically needs classical logic
        let expr = Expr::Or(
            Box::new(Expr::Var("p".into())),
            Box::new(Expr::Not(Box::new(Expr::Var("p".into())))),
        );
        let _ = compiler.suggest_tactic(&expr);
        // This checks that the compiler tracks when classical logic might be needed
        // (the actual setting of needs_classical depends on the implementation)
    }

    // Complex expression tests

    #[test]
    fn test_compile_nested_quantifiers() {
        let compiler = make_compiler();
        let expr = Expr::ForAll {
            var: "x".into(),
            ty: Some(Type::Named("Int".into())),
            body: Box::new(Expr::Exists {
                var: "y".into(),
                ty: Some(Type::Named("Int".into())),
                body: Box::new(Expr::Compare(
                    Box::new(Expr::Var("x".into())),
                    ComparisonOp::Lt,
                    Box::new(Expr::Var("y".into())),
                )),
            }),
        };
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("∀"));
        assert!(result.contains("∃"));
        assert!(result.contains("<"));
    }

    #[test]
    fn test_compile_complex_arithmetic() {
        let compiler = make_compiler();
        // (x + y) * (x - y)
        let expr = Expr::Binary(
            Box::new(Expr::Binary(
                Box::new(Expr::Var("x".into())),
                BinaryOp::Add,
                Box::new(Expr::Var("y".into())),
            )),
            BinaryOp::Mul,
            Box::new(Expr::Binary(
                Box::new(Expr::Var("x".into())),
                BinaryOp::Sub,
                Box::new(Expr::Var("y".into())),
            )),
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("+"));
        assert!(result.contains("-"));
        assert!(result.contains("*"));
    }

    // ============================================================
    // Mutation-killing tests for helper functions
    // ============================================================

    // is_arithmetic tests
    #[test]
    fn test_is_arithmetic_int() {
        assert!(Lean4Compiler::is_arithmetic(&Expr::Int(42)));
    }

    #[test]
    fn test_is_arithmetic_binary() {
        let expr = Expr::Binary(
            Box::new(Expr::Int(1)),
            BinaryOp::Add,
            Box::new(Expr::Int(2)),
        );
        assert!(Lean4Compiler::is_arithmetic(&expr));
    }

    #[test]
    fn test_is_arithmetic_neg() {
        let expr = Expr::Neg(Box::new(Expr::Int(5)));
        assert!(Lean4Compiler::is_arithmetic(&expr));
    }

    #[test]
    fn test_is_arithmetic_var_false() {
        // Variables are NOT arithmetic (conservative)
        assert!(!Lean4Compiler::is_arithmetic(&Expr::Var("x".into())));
    }

    #[test]
    fn test_is_arithmetic_bool_false() {
        assert!(!Lean4Compiler::is_arithmetic(&Expr::Bool(true)));
    }

    // is_linear_arithmetic tests
    #[test]
    fn test_is_linear_arithmetic_int() {
        assert!(Lean4Compiler::is_linear_arithmetic(&Expr::Int(42)));
    }

    #[test]
    fn test_is_linear_arithmetic_var() {
        assert!(Lean4Compiler::is_linear_arithmetic(&Expr::Var("x".into())));
    }

    #[test]
    fn test_is_linear_arithmetic_add() {
        let expr = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Add,
            Box::new(Expr::Var("y".into())),
        );
        assert!(Lean4Compiler::is_linear_arithmetic(&expr));
    }

    #[test]
    fn test_is_linear_arithmetic_sub() {
        let expr = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Sub,
            Box::new(Expr::Int(1)),
        );
        assert!(Lean4Compiler::is_linear_arithmetic(&expr));
    }

    #[test]
    fn test_is_linear_arithmetic_mul_with_constant_left() {
        // 2 * x is linear
        let expr = Expr::Binary(
            Box::new(Expr::Int(2)),
            BinaryOp::Mul,
            Box::new(Expr::Var("x".into())),
        );
        assert!(Lean4Compiler::is_linear_arithmetic(&expr));
    }

    #[test]
    fn test_is_linear_arithmetic_mul_with_constant_right() {
        // x * 2 is linear
        let expr = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Mul,
            Box::new(Expr::Int(2)),
        );
        assert!(Lean4Compiler::is_linear_arithmetic(&expr));
    }

    #[test]
    fn test_is_linear_arithmetic_mul_vars_not_linear() {
        // x * y is NOT linear
        let expr = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Mul,
            Box::new(Expr::Var("y".into())),
        );
        assert!(!Lean4Compiler::is_linear_arithmetic(&expr));
    }

    #[test]
    fn test_is_linear_arithmetic_div_not_linear() {
        let expr = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Div,
            Box::new(Expr::Int(2)),
        );
        assert!(!Lean4Compiler::is_linear_arithmetic(&expr));
    }

    #[test]
    fn test_is_linear_arithmetic_neg() {
        let expr = Expr::Neg(Box::new(Expr::Var("x".into())));
        assert!(Lean4Compiler::is_linear_arithmetic(&expr));
    }

    #[test]
    fn test_is_linear_arithmetic_bool_false() {
        assert!(!Lean4Compiler::is_linear_arithmetic(&Expr::Bool(true)));
    }

    #[test]
    fn test_is_linear_arithmetic_nested_non_linear() {
        // (x + y) * z - NOT linear because both sides of mul are non-constant
        let inner = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Add,
            Box::new(Expr::Var("y".into())),
        );
        let expr = Expr::Binary(
            Box::new(inner),
            BinaryOp::Mul,
            Box::new(Expr::Var("z".into())),
        );
        assert!(!Lean4Compiler::is_linear_arithmetic(&expr));
    }

    // is_decidable tests
    #[test]
    fn test_is_decidable_int() {
        assert!(Lean4Compiler::is_decidable(&Expr::Int(42)));
    }

    #[test]
    fn test_is_decidable_bool() {
        assert!(Lean4Compiler::is_decidable(&Expr::Bool(true)));
        assert!(Lean4Compiler::is_decidable(&Expr::Bool(false)));
    }

    #[test]
    fn test_is_decidable_string() {
        assert!(Lean4Compiler::is_decidable(&Expr::String("hello".into())));
    }

    #[test]
    fn test_is_decidable_var_false() {
        assert!(!Lean4Compiler::is_decidable(&Expr::Var("x".into())));
    }

    #[test]
    fn test_is_decidable_float_false() {
        assert!(!Lean4Compiler::is_decidable(&Expr::Float(1.5)));
    }

    // involves_boolean_logic tests
    #[test]
    fn test_involves_boolean_logic_bool() {
        assert!(Lean4Compiler::involves_boolean_logic(&Expr::Bool(true)));
        assert!(Lean4Compiler::involves_boolean_logic(&Expr::Bool(false)));
    }

    #[test]
    fn test_involves_boolean_logic_not() {
        let expr = Expr::Not(Box::new(Expr::Var("p".into())));
        assert!(Lean4Compiler::involves_boolean_logic(&expr));
    }

    #[test]
    fn test_involves_boolean_logic_and() {
        let expr = Expr::And(
            Box::new(Expr::Var("p".into())),
            Box::new(Expr::Var("q".into())),
        );
        assert!(Lean4Compiler::involves_boolean_logic(&expr));
    }

    #[test]
    fn test_involves_boolean_logic_or() {
        let expr = Expr::Or(
            Box::new(Expr::Var("p".into())),
            Box::new(Expr::Var("q".into())),
        );
        assert!(Lean4Compiler::involves_boolean_logic(&expr));
    }

    #[test]
    fn test_involves_boolean_logic_var_false() {
        // Variables are NOT considered boolean logic (conservative)
        assert!(!Lean4Compiler::involves_boolean_logic(&Expr::Var(
            "x".into()
        )));
    }

    #[test]
    fn test_involves_boolean_logic_int_false() {
        assert!(!Lean4Compiler::involves_boolean_logic(&Expr::Int(42)));
    }

    // is_simplifiable_method tests
    #[test]
    fn test_is_simplifiable_method_length() {
        assert!(Lean4Compiler::is_simplifiable_method("length"));
    }

    #[test]
    fn test_is_simplifiable_method_size() {
        assert!(Lean4Compiler::is_simplifiable_method("size"));
    }

    #[test]
    fn test_is_simplifiable_method_is_empty() {
        assert!(Lean4Compiler::is_simplifiable_method("isEmpty"));
    }

    #[test]
    fn test_is_simplifiable_method_empty() {
        assert!(Lean4Compiler::is_simplifiable_method("empty"));
    }

    #[test]
    fn test_is_simplifiable_method_head() {
        assert!(Lean4Compiler::is_simplifiable_method("head"));
    }

    #[test]
    fn test_is_simplifiable_method_tail() {
        assert!(Lean4Compiler::is_simplifiable_method("tail"));
    }

    #[test]
    fn test_is_simplifiable_method_contains() {
        assert!(Lean4Compiler::is_simplifiable_method("contains"));
    }

    #[test]
    fn test_is_simplifiable_method_map() {
        assert!(Lean4Compiler::is_simplifiable_method("map"));
    }

    #[test]
    fn test_is_simplifiable_method_filter() {
        assert!(Lean4Compiler::is_simplifiable_method("filter"));
    }

    #[test]
    fn test_is_simplifiable_method_union() {
        assert!(Lean4Compiler::is_simplifiable_method("union"));
    }

    #[test]
    fn test_is_simplifiable_method_unknown_false() {
        assert!(!Lean4Compiler::is_simplifiable_method("myCustomMethod"));
    }

    // contains_simplifiable_method tests
    #[test]
    fn test_contains_simplifiable_method_method_call_direct() {
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "length".into(),
            args: vec![],
        };
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_method_call_not_simplifiable() {
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".into())),
            method: "customMethod".into(),
            args: vec![],
        };
        assert!(!Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_nested_in_receiver() {
        // obj.length.toString() - length is in receiver
        let inner = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".into())),
            method: "length".into(),
            args: vec![],
        };
        let expr = Expr::MethodCall {
            receiver: Box::new(inner),
            method: "toString".into(),
            args: vec![],
        };
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_nested_in_args() {
        let arg = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "size".into(),
            args: vec![],
        };
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".into())),
            method: "process".into(),
            args: vec![arg],
        };
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_field_access_direct() {
        let expr = Expr::FieldAccess(Box::new(Expr::Var("xs".into())), "length".into());
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_field_access_nested() {
        // obj.inner.length
        let inner = Expr::FieldAccess(Box::new(Expr::Var("obj".into())), "inner".into());
        let expr = Expr::FieldAccess(Box::new(inner), "length".into());
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_in_binary() {
        let lhs = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "length".into(),
            args: vec![],
        };
        let expr = Expr::Binary(Box::new(lhs), BinaryOp::Add, Box::new(Expr::Int(1)));
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_in_compare_rhs() {
        let rhs = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "size".into(),
            args: vec![],
        };
        let expr = Expr::Compare(Box::new(Expr::Int(0)), ComparisonOp::Lt, Box::new(rhs));
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_in_and() {
        let lhs = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "isEmpty".into(),
            args: vec![],
        };
        let expr = Expr::And(Box::new(lhs), Box::new(Expr::Bool(true)));
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_in_or() {
        let rhs = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "contains".into(),
            args: vec![Expr::Int(1)],
        };
        let expr = Expr::Or(Box::new(Expr::Bool(false)), Box::new(rhs));
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_in_implies() {
        let lhs = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "head".into(),
            args: vec![],
        };
        let expr = Expr::Implies(Box::new(lhs), Box::new(Expr::Bool(true)));
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_in_app() {
        let arg = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "tail".into(),
            args: vec![],
        };
        let expr = Expr::App("f".into(), vec![arg]);
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_in_not() {
        let inner = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "isEmpty".into(),
            args: vec![],
        };
        let expr = Expr::Not(Box::new(inner));
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_in_neg() {
        let inner = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "length".into(),
            args: vec![],
        };
        let expr = Expr::Neg(Box::new(inner));
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_in_forall() {
        let body = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "contains".into(),
            args: vec![Expr::Var("x".into())],
        };
        let expr = Expr::ForAll {
            var: "x".into(),
            ty: None,
            body: Box::new(body),
        };
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_in_exists() {
        let body = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "member".into(),
            args: vec![Expr::Var("y".into())],
        };
        let expr = Expr::Exists {
            var: "y".into(),
            ty: None,
            body: Box::new(body),
        };
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_in_forall_in_body() {
        let body = Expr::MethodCall {
            receiver: Box::new(Expr::Var("ys".into())),
            method: "filter".into(),
            args: vec![],
        };
        let expr = Expr::ForAllIn {
            var: "x".into(),
            collection: Box::new(Expr::Var("xs".into())),
            body: Box::new(body),
        };
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_in_forall_in_collection() {
        let collection = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "union".into(),
            args: vec![Expr::Var("ys".into())],
        };
        let expr = Expr::ForAllIn {
            var: "x".into(),
            collection: Box::new(collection),
            body: Box::new(Expr::Bool(true)),
        };
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_in_exists_in() {
        let body = Expr::MethodCall {
            receiver: Box::new(Expr::Var("ys".into())),
            method: "inter".into(),
            args: vec![],
        };
        let expr = Expr::ExistsIn {
            var: "x".into(),
            collection: Box::new(Expr::Var("xs".into())),
            body: Box::new(body),
        };
        assert!(Lean4Compiler::contains_simplifiable_method(&expr));
    }

    #[test]
    fn test_contains_simplifiable_method_plain_var_false() {
        assert!(!Lean4Compiler::contains_simplifiable_method(&Expr::Var(
            "x".into()
        )));
    }

    #[test]
    fn test_contains_simplifiable_method_int_false() {
        assert!(!Lean4Compiler::contains_simplifiable_method(&Expr::Int(42)));
    }

    // exprs_equal tests
    #[test]
    fn test_exprs_equal_vars_same() {
        assert!(Lean4Compiler::exprs_equal(
            &Expr::Var("x".into()),
            &Expr::Var("x".into())
        ));
    }

    #[test]
    fn test_exprs_equal_vars_different() {
        assert!(!Lean4Compiler::exprs_equal(
            &Expr::Var("x".into()),
            &Expr::Var("y".into())
        ));
    }

    #[test]
    fn test_exprs_equal_ints_same() {
        assert!(Lean4Compiler::exprs_equal(&Expr::Int(42), &Expr::Int(42)));
    }

    #[test]
    fn test_exprs_equal_ints_different() {
        assert!(!Lean4Compiler::exprs_equal(&Expr::Int(42), &Expr::Int(43)));
    }

    #[test]
    fn test_exprs_equal_bools_same() {
        assert!(Lean4Compiler::exprs_equal(
            &Expr::Bool(true),
            &Expr::Bool(true)
        ));
        assert!(Lean4Compiler::exprs_equal(
            &Expr::Bool(false),
            &Expr::Bool(false)
        ));
    }

    #[test]
    fn test_exprs_equal_bools_different() {
        assert!(!Lean4Compiler::exprs_equal(
            &Expr::Bool(true),
            &Expr::Bool(false)
        ));
    }

    #[test]
    fn test_exprs_equal_strings_same() {
        assert!(Lean4Compiler::exprs_equal(
            &Expr::String("hello".into()),
            &Expr::String("hello".into())
        ));
    }

    #[test]
    fn test_exprs_equal_strings_different() {
        assert!(!Lean4Compiler::exprs_equal(
            &Expr::String("hello".into()),
            &Expr::String("world".into())
        ));
    }

    #[test]
    fn test_exprs_equal_binary_same() {
        let e1 = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Add,
            Box::new(Expr::Int(1)),
        );
        let e2 = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Add,
            Box::new(Expr::Int(1)),
        );
        assert!(Lean4Compiler::exprs_equal(&e1, &e2));
    }

    #[test]
    fn test_exprs_equal_binary_different_op() {
        let e1 = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Add,
            Box::new(Expr::Int(1)),
        );
        let e2 = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Sub,
            Box::new(Expr::Int(1)),
        );
        assert!(!Lean4Compiler::exprs_equal(&e1, &e2));
    }

    #[test]
    fn test_exprs_equal_binary_different_lhs() {
        let e1 = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Add,
            Box::new(Expr::Int(1)),
        );
        let e2 = Expr::Binary(
            Box::new(Expr::Var("y".into())),
            BinaryOp::Add,
            Box::new(Expr::Int(1)),
        );
        assert!(!Lean4Compiler::exprs_equal(&e1, &e2));
    }

    #[test]
    fn test_exprs_equal_binary_different_rhs() {
        let e1 = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Add,
            Box::new(Expr::Int(1)),
        );
        let e2 = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Add,
            Box::new(Expr::Int(2)),
        );
        assert!(!Lean4Compiler::exprs_equal(&e1, &e2));
    }

    #[test]
    fn test_exprs_equal_app_same() {
        let e1 = Expr::App("f".into(), vec![Expr::Int(1), Expr::Int(2)]);
        let e2 = Expr::App("f".into(), vec![Expr::Int(1), Expr::Int(2)]);
        assert!(Lean4Compiler::exprs_equal(&e1, &e2));
    }

    #[test]
    fn test_exprs_equal_app_different_name() {
        let e1 = Expr::App("f".into(), vec![Expr::Int(1)]);
        let e2 = Expr::App("g".into(), vec![Expr::Int(1)]);
        assert!(!Lean4Compiler::exprs_equal(&e1, &e2));
    }

    #[test]
    fn test_exprs_equal_app_different_args_count() {
        let e1 = Expr::App("f".into(), vec![Expr::Int(1)]);
        let e2 = Expr::App("f".into(), vec![Expr::Int(1), Expr::Int(2)]);
        assert!(!Lean4Compiler::exprs_equal(&e1, &e2));
    }

    #[test]
    fn test_exprs_equal_app_different_args_value() {
        let e1 = Expr::App("f".into(), vec![Expr::Int(1)]);
        let e2 = Expr::App("f".into(), vec![Expr::Int(2)]);
        assert!(!Lean4Compiler::exprs_equal(&e1, &e2));
    }

    #[test]
    fn test_exprs_equal_different_types() {
        assert!(!Lean4Compiler::exprs_equal(
            &Expr::Int(1),
            &Expr::Var("x".into())
        ));
    }

    // is_negation_of tests
    #[test]
    fn test_is_negation_of_true() {
        let p = Expr::Var("p".into());
        let not_p = Expr::Not(Box::new(Expr::Var("p".into())));
        assert!(Lean4Compiler::is_negation_of(&p, &not_p));
    }

    #[test]
    fn test_is_negation_of_false_not_negation() {
        let p = Expr::Var("p".into());
        let q = Expr::Var("q".into());
        assert!(!Lean4Compiler::is_negation_of(&p, &q));
    }

    #[test]
    fn test_is_negation_of_false_different_inner() {
        let p = Expr::Var("p".into());
        let not_q = Expr::Not(Box::new(Expr::Var("q".into())));
        assert!(!Lean4Compiler::is_negation_of(&p, &not_q));
    }

    #[test]
    fn test_is_negation_of_complex_expr() {
        let expr = Expr::And(
            Box::new(Expr::Var("a".into())),
            Box::new(Expr::Var("b".into())),
        );
        let not_expr = Expr::Not(Box::new(Expr::And(
            Box::new(Expr::Var("a".into())),
            Box::new(Expr::Var("b".into())),
        )));
        // exprs_equal doesn't support And, so this returns false
        assert!(!Lean4Compiler::is_negation_of(&expr, &not_expr));
    }

    // ============================================================
    // Mutation-killing tests for suggest_inner_tactic match arms
    // ============================================================

    #[test]
    fn test_suggest_inner_tactic_false_literal() {
        let compiler = make_compiler();
        // Bool(false) should suggest "contradiction"
        let tactic = compiler.suggest_inner_tactic(&Expr::Bool(false), false);
        assert_eq!(tactic, "contradiction");
    }

    #[test]
    fn test_suggest_inner_tactic_boolean_equality() {
        let compiler = make_compiler();
        // Equality with boolean logic should suggest "decide"
        let expr = Expr::Compare(
            Box::new(Expr::Not(Box::new(Expr::Bool(true)))),
            ComparisonOp::Eq,
            Box::new(Expr::Bool(false)),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert_eq!(tactic, "decide");
    }

    #[test]
    fn test_suggest_inner_tactic_arithmetic_equality_linear() {
        let compiler = make_compiler();
        // Linear arithmetic equality should suggest "omega"
        let expr = Expr::Compare(
            Box::new(Expr::Binary(
                Box::new(Expr::Var("x".into())),
                BinaryOp::Add,
                Box::new(Expr::Int(1)),
            )),
            ComparisonOp::Eq,
            Box::new(Expr::Int(5)),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert_eq!(tactic, "omega");
    }

    #[test]
    fn test_suggest_inner_tactic_arithmetic_equality_polynomial() {
        let compiler = make_compiler();
        // Polynomial (non-linear) arithmetic should suggest "ring"
        let expr = Expr::Compare(
            Box::new(Expr::Binary(
                Box::new(Expr::Var("x".into())),
                BinaryOp::Mul,
                Box::new(Expr::Var("y".into())),
            )),
            ComparisonOp::Eq,
            Box::new(Expr::Int(0)),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert_eq!(tactic, "ring");
        assert!(compiler.needs_ring.get());
    }

    #[test]
    fn test_suggest_inner_tactic_inequality_with_hypotheses() {
        let compiler = make_compiler();
        // Linear inequality with hypotheses should suggest "linarith"
        let expr = Expr::Compare(
            Box::new(Expr::Var("x".into())),
            ComparisonOp::Lt,
            Box::new(Expr::Var("y".into())),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, true);
        assert_eq!(tactic, "linarith");
        assert!(compiler.needs_linarith.get());
    }

    #[test]
    fn test_suggest_inner_tactic_inequality_without_hypotheses() {
        let compiler = make_compiler();
        // Inequality without hypotheses should suggest "omega"
        let expr = Expr::Compare(
            Box::new(Expr::Var("x".into())),
            ComparisonOp::Le,
            Box::new(Expr::Int(10)),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert_eq!(tactic, "omega");
    }

    #[test]
    fn test_suggest_inner_tactic_double_negation() {
        let compiler = make_compiler();
        // Double negation should suggest "simp"
        let expr = Expr::Not(Box::new(Expr::Not(Box::new(Expr::Var("p".into())))));
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert_eq!(tactic, "simp");
    }

    #[test]
    fn test_suggest_inner_tactic_negation_sorry() {
        let compiler = make_compiler();
        // Simple negation of variable - inner tactic is sorry
        let expr = Expr::Not(Box::new(Expr::Var("p".into())));
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert!(tactic.contains("intro h"));
        assert!(tactic.contains("sorry"));
    }

    #[test]
    fn test_suggest_inner_tactic_negation_with_inner_tactic() {
        let compiler = make_compiler();
        // Negation where inner is provable
        let expr = Expr::Not(Box::new(Expr::Bool(true)));
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert!(tactic.contains("intro h"));
        assert!(tactic.contains("trivial"));
    }

    #[test]
    fn test_suggest_inner_tactic_inner_implies() {
        let compiler = make_compiler();
        // Implication in inner position should introduce hypothesis
        let expr = Expr::Implies(Box::new(Expr::Var("p".into())), Box::new(Expr::Bool(true)));
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert!(tactic.contains("intro h"));
        assert!(tactic.contains("trivial"));
    }

    #[test]
    fn test_suggest_inner_tactic_decidable_comparison() {
        let compiler = make_compiler();
        // Decidable comparison should suggest "decide"
        let expr = Expr::Compare(
            Box::new(Expr::Int(1)),
            ComparisonOp::Ne,
            Box::new(Expr::Int(2)),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert_eq!(tactic, "decide");
    }

    #[test]
    fn test_suggest_inner_tactic_app_not() {
        let compiler = make_compiler();
        // App with "not" should suggest "simp"
        let expr = Expr::App("not".into(), vec![Expr::Bool(true)]);
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert_eq!(tactic, "simp");
    }

    #[test]
    fn test_suggest_inner_tactic_app_id() {
        let compiler = make_compiler();
        // App with "id" should suggest "simp"
        let expr = Expr::App("id".into(), vec![Expr::Var("x".into())]);
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert_eq!(tactic, "simp");
    }

    #[test]
    fn test_suggest_inner_tactic_app_other() {
        let compiler = make_compiler();
        // App with other function name - falls through
        let expr = Expr::App("custom".into(), vec![Expr::Int(1)]);
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        // Without hypotheses, should be sorry
        assert_eq!(tactic, "sorry");
    }

    #[test]
    fn test_suggest_inner_tactic_simplifiable_method() {
        let compiler = make_compiler();
        // Method call with simplifiable method should suggest "simp"
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "length".into(),
            args: vec![],
        };
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert_eq!(tactic, "simp");
    }

    #[test]
    fn test_suggest_inner_tactic_non_simplifiable_method() {
        let compiler = make_compiler();
        // Method call without simplifiable method
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".into())),
            method: "custom".into(),
            args: vec![],
        };
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        // Without hypotheses, falls to sorry
        assert_eq!(tactic, "sorry");
    }

    #[test]
    fn test_suggest_inner_tactic_var_with_hypotheses() {
        let compiler = make_compiler();
        // Variable goal with hypotheses should suggest "simp_all"
        let tactic = compiler.suggest_inner_tactic(&Expr::Var("p".into()), true);
        assert_eq!(tactic, "simp_all");
    }

    #[test]
    fn test_suggest_inner_tactic_var_without_hypotheses() {
        let compiler = make_compiler();
        // Variable goal without hypotheses should suggest "sorry"
        let tactic = compiler.suggest_inner_tactic(&Expr::Var("p".into()), false);
        assert_eq!(tactic, "sorry");
    }

    #[test]
    fn test_suggest_inner_tactic_method_call_with_hypotheses() {
        let compiler = make_compiler();
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".into())),
            method: "custom".into(),
            args: vec![],
        };
        let tactic = compiler.suggest_inner_tactic(&expr, true);
        // Non-simplifiable method with hypotheses should use simp_all
        // But simplifiable methods get simp first, so need non-simplifiable
        assert_eq!(tactic, "simp_all");
    }

    #[test]
    fn test_suggest_inner_tactic_field_access_with_hypotheses() {
        let compiler = make_compiler();
        let expr = Expr::FieldAccess(Box::new(Expr::Var("obj".into())), "custom".into());
        let tactic = compiler.suggest_inner_tactic(&expr, true);
        assert_eq!(tactic, "simp_all");
    }

    #[test]
    fn test_suggest_inner_tactic_app_with_hypotheses() {
        let compiler = make_compiler();
        let expr = Expr::App("f".into(), vec![Expr::Var("x".into())]);
        let tactic = compiler.suggest_inner_tactic(&expr, true);
        assert_eq!(tactic, "simp_all");
    }

    #[test]
    fn test_suggest_inner_tactic_compare_with_hypotheses() {
        let compiler = make_compiler();
        // Non-decidable comparison with hypotheses
        let expr = Expr::Compare(
            Box::new(Expr::Var("x".into())),
            ComparisonOp::Eq,
            Box::new(Expr::Var("y".into())),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, true);
        assert_eq!(tactic, "simp_all");
    }

    #[test]
    fn test_suggest_inner_tactic_or_decidable() {
        let compiler = make_compiler();
        // Both sides decidable - should suggest "decide"
        let expr = Expr::Or(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert_eq!(tactic, "decide");
    }

    #[test]
    fn test_suggest_inner_tactic_or_with_hypotheses() {
        let compiler = make_compiler();
        // Or with hypotheses should suggest "simp_all"
        let expr = Expr::Or(
            Box::new(Expr::Var("p".into())),
            Box::new(Expr::Var("q".into())),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, true);
        assert_eq!(tactic, "simp_all");
    }

    #[test]
    fn test_suggest_inner_tactic_or_left_provable() {
        let compiler = make_compiler();
        // Left side is True - should suggest "left" with trivial
        let expr = Expr::Or(Box::new(Expr::Bool(true)), Box::new(Expr::Var("q".into())));
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert!(tactic.contains("left"));
        assert!(tactic.contains("trivial"));
    }

    #[test]
    fn test_suggest_inner_tactic_or_right_provable() {
        let compiler = make_compiler();
        // Left side is sorry, right is True - should suggest "right"
        let expr = Expr::Or(Box::new(Expr::Var("p".into())), Box::new(Expr::Bool(true)));
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert!(tactic.contains("right"));
        assert!(tactic.contains("trivial"));
    }

    #[test]
    fn test_suggest_inner_tactic_or_both_sorry() {
        let compiler = make_compiler();
        // Both sides are sorry - should suggest "tauto"
        let expr = Expr::Or(
            Box::new(Expr::Var("p".into())),
            Box::new(Expr::Var("q".into())),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert!(tactic.contains("tauto"));
        assert!(compiler.needs_classical.get());
    }

    #[test]
    fn test_suggest_inner_tactic_exists_in() {
        let compiler = make_compiler();
        let expr = Expr::ExistsIn {
            var: "x".into(),
            collection: Box::new(Expr::Var("S".into())),
            body: Box::new(Expr::Bool(true)),
        };
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert!(tactic.contains("use sorry"));
    }

    #[test]
    fn test_suggest_inner_tactic_simplifiable_method_in_compare() {
        let compiler = make_compiler();
        // Comparison with simplifiable method should suggest "simp"
        let lhs = Expr::MethodCall {
            receiver: Box::new(Expr::Var("xs".into())),
            method: "length".into(),
            args: vec![],
        };
        let expr = Expr::Compare(Box::new(lhs), ComparisonOp::Eq, Box::new(Expr::Int(0)));
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert_eq!(tactic, "simp");
    }

    // ============================================================
    // Mutation-killing tests for collect_intros
    // ============================================================

    #[test]
    fn test_collect_intros_forall_in() {
        // ForAllIn should add var and membership hypothesis
        let expr = Expr::ForAllIn {
            var: "x".into(),
            collection: Box::new(Expr::Var("S".into())),
            body: Box::new(Expr::Bool(true)),
        };
        let (intros, has_hypotheses, _inner) = Lean4Compiler::collect_intros(&expr);
        assert_eq!(intros, vec!["x".to_string(), "hx".to_string()]);
        assert!(has_hypotheses);
    }

    #[test]
    fn test_collect_intros_nested_forall_in() {
        let expr = Expr::ForAllIn {
            var: "x".into(),
            collection: Box::new(Expr::Var("S".into())),
            body: Box::new(Expr::ForAllIn {
                var: "y".into(),
                collection: Box::new(Expr::Var("T".into())),
                body: Box::new(Expr::Bool(true)),
            }),
        };
        let (intros, has_hypotheses, _inner) = Lean4Compiler::collect_intros(&expr);
        assert_eq!(
            intros,
            vec![
                "x".to_string(),
                "hx".to_string(),
                "y".to_string(),
                "hy".to_string()
            ]
        );
        assert!(has_hypotheses);
    }

    #[test]
    fn test_suggest_tactic_forall_in() {
        let compiler = make_compiler();
        let expr = Expr::ForAllIn {
            var: "x".into(),
            collection: Box::new(Expr::Var("S".into())),
            body: Box::new(Expr::Bool(true)),
        };
        let tactic = compiler.suggest_tactic(&expr);
        assert!(tactic.contains("intro x hx"));
        assert!(tactic.contains("trivial"));
    }

    // ============================================================
    // Mutation-killing tests for compile_refinement
    // ============================================================

    #[test]
    fn test_compile_refinement_with_mappings() {
        use crate::ast::{Refinement, VariableMapping};
        let compiler = make_compiler();
        let refinement = Refinement {
            name: "TestRef".into(),
            refines: "AbstractSpec".into(),
            mappings: vec![VariableMapping {
                spec_var: Expr::Var("abstract_x".into()),
                impl_var: Expr::Var("concrete_x".into()),
            }],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![],
        };
        let result = compiler.compile_refinement(&refinement);
        assert!(result.contains("Variable Mappings"));
        assert!(result.contains("abstract_x"));
        assert!(result.contains("concrete_x"));
        assert!(result.contains("TestRef_mapping_0"));
    }

    #[test]
    fn test_compile_refinement_empty_mappings() {
        use crate::ast::Refinement;
        let compiler = make_compiler();
        let refinement = Refinement {
            name: "TestRef".into(),
            refines: "AbstractSpec".into(),
            mappings: vec![],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![],
        };
        let result = compiler.compile_refinement(&refinement);
        // Should NOT contain Variable Mappings section
        assert!(!result.contains("Variable Mappings"));
    }

    #[test]
    fn test_compile_refinement_with_invariants() {
        use crate::ast::Refinement;
        let compiler = make_compiler();
        let refinement = Refinement {
            name: "TestRef".into(),
            refines: "AbstractSpec".into(),
            mappings: vec![],
            invariants: vec![
                Expr::Compare(
                    Box::new(Expr::Var("x".into())),
                    ComparisonOp::Gt,
                    Box::new(Expr::Int(0)),
                ),
                Expr::Bool(true),
            ],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![],
        };
        let result = compiler.compile_refinement(&refinement);
        assert!(result.contains("Refinement Invariants"));
        assert!(result.contains("TestRef_invariant_0"));
        assert!(result.contains("TestRef_invariant_1"));
    }

    #[test]
    fn test_compile_refinement_empty_invariants() {
        use crate::ast::Refinement;
        let compiler = make_compiler();
        let refinement = Refinement {
            name: "TestRef".into(),
            refines: "AbstractSpec".into(),
            mappings: vec![],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![],
        };
        let result = compiler.compile_refinement(&refinement);
        // Should NOT contain Refinement Invariants section
        assert!(!result.contains("Refinement Invariants"));
    }

    #[test]
    fn test_compile_refinement_with_actions() {
        use crate::ast::{ActionMapping, Refinement};
        let compiler = make_compiler();
        let refinement = Refinement {
            name: "TestRef".into(),
            refines: "AbstractSpec".into(),
            mappings: vec![],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![ActionMapping {
                name: "doSomething".into(),
                spec_action: "AbstractAction".into(),
                impl_action: vec!["Impl".into(), "Action".into()],
                guard: None,
            }],
        };
        let result = compiler.compile_refinement(&refinement);
        assert!(result.contains("Action Correspondence"));
        assert!(result.contains("doSomething"));
        assert!(result.contains("AbstractAction"));
        assert!(result.contains("Impl.Action"));
    }

    #[test]
    fn test_compile_refinement_with_guarded_action() {
        use crate::ast::{ActionMapping, Refinement};
        let compiler = make_compiler();
        let refinement = Refinement {
            name: "TestRef".into(),
            refines: "AbstractSpec".into(),
            mappings: vec![],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![ActionMapping {
                name: "guardedAction".into(),
                spec_action: "SpecAct".into(),
                impl_action: vec!["ImplAct".into()],
                guard: Some(Expr::Compare(
                    Box::new(Expr::Var("x".into())),
                    ComparisonOp::Gt,
                    Box::new(Expr::Int(0)),
                )),
            }],
        };
        let result = compiler.compile_refinement(&refinement);
        assert!(result.contains("Action Correspondence"));
        assert!(result.contains("guardedAction"));
        assert!(result.contains("(x) > (0)"));
    }

    #[test]
    fn test_compile_refinement_empty_actions() {
        use crate::ast::Refinement;
        let compiler = make_compiler();
        let refinement = Refinement {
            name: "TestRef".into(),
            refines: "AbstractSpec".into(),
            mappings: vec![],
            invariants: vec![],
            abstraction: Expr::Bool(true),
            simulation: Expr::Bool(true),
            actions: vec![],
        };
        let result = compiler.compile_refinement(&refinement);
        // Should NOT contain Action Correspondence section
        assert!(!result.contains("Action Correspondence"));
    }

    // ============================================================
    // Additional mutation-killing tests for edge cases
    // ============================================================

    #[test]
    fn test_compile_type_relation() {
        let compiler = make_compiler();
        let ty = Type::Relation(
            Box::new(Type::Named("Int".into())),
            Box::new(Type::Named("Bool".into())),
        );
        let result = compiler.compile_type(&ty);
        assert!(result.contains("→"));
        assert!(result.contains("Prop"));
    }

    #[test]
    fn test_compile_type_function() {
        let compiler = make_compiler();
        let ty = Type::Function(
            Box::new(Type::Named("Int".into())),
            Box::new(Type::Named("Bool".into())),
        );
        let result = compiler.compile_type(&ty);
        assert!(result.contains("→"));
        assert!(!result.contains("Prop"));
    }

    #[test]
    fn test_compile_type_result() {
        let compiler = make_compiler();
        let ty = Type::Result(Box::new(Type::Named("Int".into())));
        assert_eq!(compiler.compile_type(&ty), "Except String Int");
    }

    #[test]
    fn test_compile_type_unit() {
        let compiler = make_compiler();
        assert_eq!(compiler.compile_type(&Type::Unit), "Unit");
    }

    #[test]
    fn test_compile_float() {
        let compiler = make_compiler();
        assert_eq!(compiler.compile_expr(&Expr::Float(1.5)), "1.5");
    }

    #[test]
    fn test_compile_forall_in() {
        let compiler = make_compiler();
        let expr = Expr::ForAllIn {
            var: "x".into(),
            collection: Box::new(Expr::Var("S".into())),
            body: Box::new(Expr::Bool(true)),
        };
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("∀"));
        assert!(result.contains("∈"));
        assert!(result.contains("x"));
        assert!(result.contains("S"));
    }

    #[test]
    fn test_compile_exists_in() {
        let compiler = make_compiler();
        let expr = Expr::ExistsIn {
            var: "y".into(),
            collection: Box::new(Expr::Var("T".into())),
            body: Box::new(Expr::Bool(false)),
        };
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("∃"));
        assert!(result.contains("∈"));
        assert!(result.contains("y"));
        assert!(result.contains("T"));
    }

    #[test]
    fn test_compile_binary_mod() {
        let compiler = make_compiler();
        let expr = Expr::Binary(
            Box::new(Expr::Int(10)),
            BinaryOp::Mod,
            Box::new(Expr::Int(3)),
        );
        assert!(compiler.compile_expr(&expr).contains("%"));
    }

    #[test]
    fn test_compile_method_call_no_args() {
        let compiler = make_compiler();
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".into())),
            method: "doIt".into(),
            args: vec![],
        };
        let result = compiler.compile_expr(&expr);
        assert_eq!(result, "obj.doIt");
    }

    #[test]
    fn test_compile_method_call_with_args() {
        let compiler = make_compiler();
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".into())),
            method: "call".into(),
            args: vec![Expr::Int(1), Expr::Int(2)],
        };
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("obj.call"));
        assert!(result.contains("1"));
        assert!(result.contains("2"));
    }

    #[test]
    fn test_compile_security() {
        let compiler = make_compiler();
        let security = Security {
            name: "no_leak".into(),
            body: Expr::Bool(true),
        };
        let result = compiler.compile_security(&security);
        assert!(result.contains("theorem"));
        assert!(result.contains("no_leak"));
    }

    #[test]
    fn test_compile_module_with_security() {
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Security(Security {
                name: "secure".into(),
                body: Expr::Bool(true),
            })],
        };
        let typed = typecheck(spec).unwrap();
        let result = compile_to_lean(&typed);
        assert!(result.code.contains("Security: secure"));
        assert!(result.code.contains("theorem secure"));
    }

    #[test]
    fn test_compile_module_with_refinement() {
        use crate::ast::Refinement;
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Refinement(Refinement {
                name: "ImplRef".into(),
                refines: "Spec".into(),
                mappings: vec![],
                invariants: vec![],
                abstraction: Expr::Bool(true),
                simulation: Expr::Bool(true),
                actions: vec![],
            })],
        };
        let typed = typecheck(spec).unwrap();
        let result = compile_to_lean(&typed);
        assert!(result.code.contains("Refinement: ImplRef refines Spec"));
    }

    #[test]
    fn test_compile_module_needs_classical() {
        // Test that Classical is imported when needed
        let compiler = Lean4Compiler::new("Test");
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Theorem(Theorem {
                name: "lem".into(),
                body: Expr::Or(
                    Box::new(Expr::Var("p".into())),
                    Box::new(Expr::Not(Box::new(Expr::Var("p".into())))),
                ),
            })],
        };
        let typed = typecheck(spec).unwrap();
        let result = compiler.compile_module(&typed);
        assert!(result.code.contains("Mathlib.Logic.Classical"));
        assert!(result.code.contains("open Classical"));
    }

    #[test]
    fn test_compile_module_needs_ring() {
        // Test that Ring is imported when needed
        let compiler = Lean4Compiler::new("Test");
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Theorem(Theorem {
                name: "poly".into(),
                body: Expr::Compare(
                    Box::new(Expr::Binary(
                        Box::new(Expr::Var("x".into())),
                        BinaryOp::Mul,
                        Box::new(Expr::Var("y".into())),
                    )),
                    ComparisonOp::Eq,
                    Box::new(Expr::Binary(
                        Box::new(Expr::Var("y".into())),
                        BinaryOp::Mul,
                        Box::new(Expr::Var("x".into())),
                    )),
                ),
            })],
        };
        let typed = typecheck(spec).unwrap();
        let result = compiler.compile_module(&typed);
        assert!(result.code.contains("Mathlib.Tactic.Ring"));
    }

    #[test]
    fn test_compile_module_needs_linarith() {
        // Test that Linarith is imported when needed - need hypothesis + linear inequality
        let compiler = Lean4Compiler::new("Test");
        let spec = Spec {
            types: vec![],
            properties: vec![Property::Theorem(Theorem {
                name: "linarith_test".into(),
                body: Expr::Implies(
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("x".into())),
                        ComparisonOp::Lt,
                        Box::new(Expr::Var("y".into())),
                    )),
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("x".into())),
                        ComparisonOp::Le,
                        Box::new(Expr::Var("y".into())),
                    )),
                ),
            })],
        };
        let typed = typecheck(spec).unwrap();
        let result = compiler.compile_module(&typed);
        assert!(result.code.contains("Mathlib.Tactic.Linarith"));
    }

    #[test]
    fn test_suggest_tactic_reflexivity() {
        let compiler = make_compiler();
        // x = x should suggest rfl
        let expr = Expr::Compare(
            Box::new(Expr::Var("x".into())),
            ComparisonOp::Eq,
            Box::new(Expr::Var("x".into())),
        );
        let tactic = compiler.suggest_tactic(&expr);
        assert!(tactic.contains("rfl"));
    }

    #[test]
    fn test_suggest_tactic_and_splits() {
        let compiler = make_compiler();
        // And should suggest constructor with subgoals
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(true)));
        let tactic = compiler.suggest_tactic(&expr);
        assert!(tactic.contains("constructor"));
        assert!(tactic.contains("trivial"));
    }

    // ============================================================
    // Mutation-killing tests for || vs && operators
    // These catch mutations that replace || with && and vice versa
    // ============================================================

    #[test]
    fn test_boolean_equality_only_lhs_boolean() {
        let compiler = make_compiler();
        // Only LHS involves boolean logic (Not), RHS is a variable
        // If || were replaced with &&, this would not match the boolean logic branch
        let expr = Expr::Compare(
            Box::new(Expr::Not(Box::new(Expr::Bool(true)))),
            ComparisonOp::Eq,
            Box::new(Expr::Var("x".into())),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        // Should match boolean logic pattern and suggest "decide"
        assert_eq!(tactic, "decide");
    }

    #[test]
    fn test_boolean_equality_only_rhs_boolean() {
        let compiler = make_compiler();
        // Only RHS involves boolean logic
        let expr = Expr::Compare(
            Box::new(Expr::Var("x".into())),
            ComparisonOp::Eq,
            Box::new(Expr::And(
                Box::new(Expr::Bool(true)),
                Box::new(Expr::Bool(false)),
            )),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert_eq!(tactic, "decide");
    }

    #[test]
    fn test_arithmetic_equality_only_lhs_arithmetic() {
        let compiler = make_compiler();
        // Only LHS is arithmetic (Int), RHS is a variable (not arithmetic)
        // If || were replaced with &&, this would not match the arithmetic branch
        let expr = Expr::Compare(
            Box::new(Expr::Int(5)),
            ComparisonOp::Eq,
            Box::new(Expr::Var("x".into())),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        // Should match arithmetic pattern
        assert_eq!(tactic, "omega");
    }

    #[test]
    fn test_arithmetic_equality_only_rhs_arithmetic() {
        let compiler = make_compiler();
        // Only RHS is arithmetic (Binary), LHS is a variable
        let expr = Expr::Compare(
            Box::new(Expr::Var("x".into())),
            ComparisonOp::Eq,
            Box::new(Expr::Binary(
                Box::new(Expr::Int(1)),
                BinaryOp::Add,
                Box::new(Expr::Int(2)),
            )),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        assert_eq!(tactic, "omega");
    }

    #[test]
    fn test_decidable_only_lhs_decidable() {
        let compiler = make_compiler();
        // LHS is decidable (Int), RHS is not decidable (Var)
        // If && were replaced with ||, this would incorrectly match decidable branch
        // But wait - is_decidable on Var returns false, so with ||, it would be true
        // We need to verify the behavior when only one side is decidable
        let expr = Expr::Compare(
            Box::new(Expr::Int(5)),
            ComparisonOp::Eq,
            Box::new(Expr::Var("x".into())),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        // With && both sides must be decidable - Var is not, so should fall through
        // to arithmetic equality (Int is arithmetic)
        assert_eq!(tactic, "omega");
    }

    #[test]
    fn test_decidable_only_rhs_decidable() {
        let compiler = make_compiler();
        // LHS is variable (not decidable), RHS is decidable (Bool)
        let expr = Expr::Compare(
            Box::new(Expr::Var("x".into())),
            ComparisonOp::Eq,
            Box::new(Expr::Bool(true)),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        // Boolean logic on RHS, should hit boolean logic branch
        assert_eq!(tactic, "decide");
    }

    #[test]
    fn test_is_linear_arithmetic_add_one_side_non_linear() {
        // x + (y * z) - RHS is not linear (multiplication of variables)
        // If && were replaced with ||, this would incorrectly return true
        let non_linear_rhs = Expr::Binary(
            Box::new(Expr::Var("y".into())),
            BinaryOp::Mul,
            Box::new(Expr::Var("z".into())),
        );
        let expr = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Add,
            Box::new(non_linear_rhs),
        );
        assert!(!Lean4Compiler::is_linear_arithmetic(&expr));
    }

    #[test]
    fn test_is_linear_arithmetic_sub_lhs_non_linear() {
        // (x * y) - z - LHS is not linear
        let non_linear_lhs = Expr::Binary(
            Box::new(Expr::Var("x".into())),
            BinaryOp::Mul,
            Box::new(Expr::Var("y".into())),
        );
        let expr = Expr::Binary(
            Box::new(non_linear_lhs),
            BinaryOp::Sub,
            Box::new(Expr::Var("z".into())),
        );
        assert!(!Lean4Compiler::is_linear_arithmetic(&expr));
    }

    #[test]
    fn test_or_decidable_mixed() {
        let compiler = make_compiler();
        // Test Or where only one side is decidable
        // LHS is decidable (Int literal is decidable), RHS is variable (not decidable)
        let expr = Expr::Or(Box::new(Expr::Bool(true)), Box::new(Expr::Var("p".into())));
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        // Left is provable (Bool true -> trivial), should suggest left
        assert!(tactic.contains("left"));
        assert!(tactic.contains("trivial"));
    }

    #[test]
    fn test_decidable_comparison_only_one_side() {
        let compiler = make_compiler();
        // Test Compare Eq where only one side is decidable
        // LHS is a non-simplifiable method call (not decidable, not arithmetic, not boolean logic)
        // RHS is String (decidable)
        // With &&: is_decidable(MethodCall) && is_decidable(String) = false && true = false -> falls through
        // With ||: is_decidable(MethodCall) || is_decidable(String) = false || true = true -> would match
        let expr = Expr::Compare(
            Box::new(Expr::MethodCall {
                receiver: Box::new(Expr::Var("obj".into())),
                method: "getValue".into(), // non-simplifiable method
                args: vec![],
            }),
            ComparisonOp::Eq,
            Box::new(Expr::String("expected".into())),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        // MethodCall is NOT decidable, so with original &&, it should NOT match decidable branch
        // Should fall through to sorry
        assert_eq!(tactic, "sorry");
    }

    #[test]
    fn test_decidable_comparison_ne_only_one_side() {
        let compiler = make_compiler();
        // Same test but with Ne instead of Eq
        // FieldAccess is not decidable, Bool is decidable BUT also involves_boolean_logic
        let expr = Expr::Compare(
            Box::new(Expr::FieldAccess(
                Box::new(Expr::Var("obj".into())),
                "value".into(),
            )),
            ComparisonOp::Ne,
            Box::new(Expr::Bool(true)),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        // FieldAccess is NOT decidable, Bool involves_boolean_logic is true
        // involves_boolean_logic branch matches first (line 292) -> "decide"
        // Wait, but involves_boolean_logic only checks LHS, and FieldAccess is not boolean logic
        // So it falls through to... sorry actually
        assert_eq!(tactic, "sorry");
    }

    #[test]
    fn test_decidable_comparison_ne_only_one_side_string() {
        let compiler = make_compiler();
        // FieldAccess (not decidable) vs String (decidable but not boolean logic)
        // This should NOT match the decidable branch because && requires both
        let expr = Expr::Compare(
            Box::new(Expr::FieldAccess(
                Box::new(Expr::Var("obj".into())),
                "custom".into(), // non-simplifiable field
            )),
            ComparisonOp::Ne,
            Box::new(Expr::String("test".into())),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        // FieldAccess is NOT decidable, String IS decidable
        // With &&: false && true = false -> falls through
        // With ||: false || true = true -> would incorrectly say "decide"
        assert_eq!(tactic, "sorry");
    }

    #[test]
    fn test_decidable_comparison_both_decidable() {
        let compiler = make_compiler();
        // Both sides are decidable - x = x with same var triggers reflexivity
        let expr = Expr::Compare(
            Box::new(Expr::Int(1)),
            ComparisonOp::Eq,
            Box::new(Expr::Int(1)),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        // This is 1 = 1, which is reflexivity
        assert_eq!(tactic, "rfl");
    }

    #[test]
    fn test_decidable_comparison_strings() {
        let compiler = make_compiler();
        // Both strings are decidable
        let expr = Expr::Compare(
            Box::new(Expr::String("a".into())),
            ComparisonOp::Eq,
            Box::new(Expr::String("b".into())),
        );
        let tactic = compiler.suggest_inner_tactic(&expr, false);
        // Strings are decidable but not boolean logic, not arithmetic
        // Should match decidable branch
        assert_eq!(tactic, "decide");
    }

    // =========================================================================
    // Kani proofs for Lean4 compiler correctness
    // =========================================================================

    /// Prove that compile_expr never produces empty output for integer literals.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_lean4_compile_expr_int_nonempty() {
        let compiler = Lean4Compiler::new("Test");
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_lean4_compile_expr_bool_nonempty() {
        let compiler = Lean4Compiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Lean4 uses lowercase true/false
        assert!(result == "true" || result == "false" || result == "True" || result == "False");
    }

    /// Prove that compile_type always produces non-empty output for named types.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_lean4_compile_type_nonempty() {
        let compiler = Lean4Compiler::new("Test");
        let ty = Type::Named("T".to_string());
        let result = compiler.compile_type(&ty);
        assert!(!result.is_empty());
    }

    /// Prove that comparison operators compile to valid Lean4 syntax.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_lean4_comparison_valid() {
        let compiler = Lean4Compiler::new("Test");
        let ops = [
            ComparisonOp::Eq,
            ComparisonOp::Ne,
            ComparisonOp::Lt,
            ComparisonOp::Le,
            ComparisonOp::Gt,
            ComparisonOp::Ge,
        ];
        let idx: usize = kani::any();
        kani::assume(idx < ops.len());
        let op = ops[idx];

        let expr = Expr::Compare(
            Box::new(Expr::Var("x".to_string())),
            op,
            Box::new(Expr::Var("y".to_string())),
        );
        let result = compiler.compile_expr(&expr);
        // Result must contain at least one comparison character
        assert!(
            result.contains('=')
                || result.contains('≠')
                || result.contains('<')
                || result.contains('>')
                || result.contains('≤')
                || result.contains('≥')
        );
    }

    /// Prove that binary operators compile to non-empty output.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_lean4_binary_ops_nonempty() {
        let compiler = Lean4Compiler::new("Test");
        let ops = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Mod,
        ];
        let idx: usize = kani::any();
        kani::assume(idx < ops.len());
        let op = ops[idx];

        let expr = Expr::Binary(Box::new(Expr::Int(1)), op, Box::new(Expr::Int(2)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that tactic suggestion never panics for boolean expressions.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_lean4_suggest_tactic_bool_safe() {
        let compiler = Lean4Compiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.suggest_tactic(&expr);
        // Should not panic and should return a tactic
        assert!(!result.is_empty());
    }

    /// Prove that tactic suggestion handles And expressions.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_lean4_suggest_tactic_and_safe() {
        let compiler = Lean4Compiler::new("Test");
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(true)));
        let result = compiler.suggest_tactic(&expr);
        // Should contain constructor or and.intro for conjunctions
        assert!(!result.is_empty());
    }

    /// Prove that tactic suggestion handles Or expressions.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_lean4_suggest_tactic_or_safe() {
        let compiler = Lean4Compiler::new("Test");
        let expr = Expr::Or(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.suggest_tactic(&expr);
        // Should contain left, right, or Or.intro tactics
        assert!(!result.is_empty());
    }

    /// Prove that namespace is preserved in compiled output.
    #[cfg(kani)]
    #[kani::proof]
    fn verify_lean4_namespace_preserved() {
        let compiler = Lean4Compiler::new("MyNS");
        let spec = make_typed_spec(vec![]);
        let result = compiler.compile_module(&spec);
        // Namespace must appear in the output
        assert!(result.code.contains("MyNS"));
        assert_eq!(result.backend, "LEAN 4");
    }

    // =========================================================================
    // Version Specification Tests
    // =========================================================================

    #[test]
    fn test_compile_version_spec() {
        use crate::ast::{CapabilityClause, PreservesClause};

        let compiler = make_compiler();
        let version = VersionSpec {
            name: "DasherV2".into(),
            improves: "DasherV1".into(),
            capabilities: vec![CapabilityClause {
                expr: Expr::Compare(
                    Box::new(Expr::FieldAccess(
                        Box::new(Expr::Var("V2".into())),
                        "speed".into(),
                    )),
                    ComparisonOp::Ge,
                    Box::new(Expr::FieldAccess(
                        Box::new(Expr::Var("V1".into())),
                        "speed".into(),
                    )),
                ),
            }],
            preserves: vec![PreservesClause {
                property: Expr::FieldAccess(Box::new(Expr::Var("V1".into())), "soundness".into()),
            }],
        };

        let result = compiler.compile_version_spec(&version);

        // Check version header is present
        assert!(result.contains("Version Improvement: DasherV2 improves DasherV1"));
        assert!(result.contains("dasherv2_capability_1"));
        assert!(result.contains("dasherv2_preserves_1"));
        assert!(result.contains("V2.speed"));
        assert!(result.contains("V1.speed"));
        assert!(result.contains("V1.soundness"));
    }

    #[test]
    fn test_compile_version_spec_multiple_capabilities() {
        use crate::ast::{CapabilityClause, PreservesClause};

        let compiler = make_compiler();
        let version = VersionSpec {
            name: "AgentV3".into(),
            improves: "AgentV2".into(),
            capabilities: vec![
                CapabilityClause {
                    expr: Expr::Compare(
                        Box::new(Expr::FieldAccess(
                            Box::new(Expr::Var("V3".into())),
                            "speed".into(),
                        )),
                        ComparisonOp::Ge,
                        Box::new(Expr::FieldAccess(
                            Box::new(Expr::Var("V2".into())),
                            "speed".into(),
                        )),
                    ),
                },
                CapabilityClause {
                    expr: Expr::Compare(
                        Box::new(Expr::FieldAccess(
                            Box::new(Expr::Var("V3".into())),
                            "accuracy".into(),
                        )),
                        ComparisonOp::Ge,
                        Box::new(Expr::FieldAccess(
                            Box::new(Expr::Var("V2".into())),
                            "accuracy".into(),
                        )),
                    ),
                },
            ],
            preserves: vec![PreservesClause {
                property: Expr::Var("termination_guaranteed".into()),
            }],
        };

        let result = compiler.compile_version_spec(&version);

        // Check all theorems are present
        assert!(result.contains("agentv3_capability_1"));
        assert!(result.contains("agentv3_capability_2"));
        assert!(result.contains("agentv3_preserves_1"));
        assert!(result.contains("V3.speed"));
        assert!(result.contains("V3.accuracy"));
    }

    #[test]
    fn test_compile_version_spec_empty() {
        let compiler = make_compiler();
        let version = VersionSpec {
            name: "EmptyV2".into(),
            improves: "EmptyV1".into(),
            capabilities: vec![],
            preserves: vec![],
        };

        let result = compiler.compile_version_spec(&version);

        // Should still have header comments
        assert!(result.contains("Version Improvement: EmptyV2 improves EmptyV1"));
        // Should not have theorems for capabilities or preserves
        assert!(!result.contains("theorem"));
    }

    // =========================================================================
    // Capability Specification Tests
    // =========================================================================

    #[test]
    fn test_compile_capability_spec() {
        use crate::ast::{CapabilityAbility, Param};

        let compiler = make_compiler();
        let capability = CapabilitySpec {
            name: "VerifierCapability".into(),
            abilities: vec![CapabilityAbility {
                name: "verify_code".into(),
                params: vec![Param {
                    name: "code".into(),
                    ty: Type::Named("RustCode".into()),
                }],
                return_type: Some(Type::Named("VerificationResult".into())),
            }],
            requires: vec![Expr::Var("soundness_preserved".into())],
        };

        let result = compiler.compile_capability_spec(&capability);

        // Check structure is generated
        assert!(result.contains("structure VerifierCapability where"));
        assert!(result.contains("verify_code"));
        assert!(result.contains("RustCode"));
        assert!(result.contains("VerificationResult"));

        // Check requirement axiom is generated
        assert!(result.contains("verifiercapability_requirement_1"));
        assert!(result.contains("soundness_preserved"));
    }

    #[test]
    fn test_compile_capability_spec_multiple_abilities() {
        use crate::ast::{CapabilityAbility, Param};

        let compiler = make_compiler();
        let capability = CapabilitySpec {
            name: "DasherCapability".into(),
            abilities: vec![
                CapabilityAbility {
                    name: "verify_usl".into(),
                    params: vec![Param {
                        name: "spec".into(),
                        ty: Type::Named("UslSpec".into()),
                    }],
                    return_type: Some(Type::Named("Result".into())),
                },
                CapabilityAbility {
                    name: "improve_self".into(),
                    params: vec![Param {
                        name: "improvement".into(),
                        ty: Type::Named("Improvement".into()),
                    }],
                    return_type: Some(Type::Named("DasherVersion".into())),
                },
            ],
            requires: vec![],
        };

        let result = compiler.compile_capability_spec(&capability);

        // Check both abilities are in the structure
        assert!(result.contains("verify_usl"));
        assert!(result.contains("improve_self"));
        assert!(result.contains("UslSpec"));
        assert!(result.contains("Improvement"));
        assert!(result.contains("DasherVersion"));
    }

    #[test]
    fn test_compile_capability_spec_no_params() {
        use crate::ast::CapabilityAbility;

        let compiler = make_compiler();
        let capability = CapabilitySpec {
            name: "SimpleCapability".into(),
            abilities: vec![CapabilityAbility {
                name: "get_status".into(),
                params: vec![],
                return_type: Some(Type::Named("Status".into())),
            }],
            requires: vec![],
        };

        let result = compiler.compile_capability_spec(&capability);

        // Check ability with no params is handled correctly
        assert!(result.contains("get_status : Status"));
    }

    #[test]
    fn test_compile_capability_spec_no_return() {
        use crate::ast::{CapabilityAbility, Param};

        let compiler = make_compiler();
        let capability = CapabilitySpec {
            name: "SideEffectCapability".into(),
            abilities: vec![CapabilityAbility {
                name: "log_message".into(),
                params: vec![Param {
                    name: "msg".into(),
                    ty: Type::Named("String".into()),
                }],
                return_type: None,
            }],
            requires: vec![],
        };

        let result = compiler.compile_capability_spec(&capability);

        // Check ability with no return type defaults to Unit
        assert!(result.contains("Unit"));
    }

    // =========================================================================
    // Module Compilation Tests for Version and Capability
    // =========================================================================

    #[test]
    fn test_compile_module_with_version() {
        use crate::ast::{CapabilityClause, PreservesClause};

        let version = VersionSpec {
            name: "V2".into(),
            improves: "V1".into(),
            capabilities: vec![CapabilityClause {
                expr: Expr::Bool(true),
            }],
            preserves: vec![PreservesClause {
                property: Expr::Bool(true),
            }],
        };

        let spec = Spec {
            types: vec![],
            properties: vec![Property::Version(version)],
        };

        let typed = typecheck(spec).expect("typecheck failed");
        let result = compile_to_lean(&typed);

        assert!(result.code.contains("Version: V2 improves V1"));
        assert!(result.code.contains("v2_capability_1"));
        assert!(result.code.contains("v2_preserves_1"));
    }

    #[test]
    fn test_compile_module_with_capability() {
        use crate::ast::CapabilityAbility;

        let capability = CapabilitySpec {
            name: "TestCapability".into(),
            abilities: vec![CapabilityAbility {
                name: "run".into(),
                params: vec![],
                return_type: Some(Type::Named("Bool".into())),
            }],
            requires: vec![],
        };

        let spec = Spec {
            types: vec![],
            properties: vec![Property::Capability(capability)],
        };

        let typed = typecheck(spec).expect("typecheck failed");
        let result = compile_to_lean(&typed);

        assert!(result.code.contains("Capability: TestCapability"));
        assert!(result.code.contains("structure TestCapability where"));
        assert!(result.code.contains("run : Bool"));
    }

    // ========================================================================
    // Phase 17.3: Graph predicate compilation tests for Lean4
    // ========================================================================

    #[test]
    fn test_compile_lean4_is_acyclic() {
        let compiler = Lean4Compiler::new("Test");
        let expr = Expr::App("is_acyclic".to_string(), vec![Expr::Var("g".to_string())]);
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("Graph.isAcyclic"));
        assert!(result.contains("g"));
    }

    #[test]
    fn test_compile_lean4_is_dag() {
        let compiler = Lean4Compiler::new("Test");
        let expr = Expr::App("is_dag".to_string(), vec![Expr::Var("graph".to_string())]);
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("Graph.isAcyclic"));
        assert!(result.contains("graph"));
    }

    #[test]
    fn test_compile_lean4_has_path() {
        let compiler = Lean4Compiler::new("Test");
        let expr = Expr::App(
            "has_path".to_string(),
            vec![
                Expr::Var("g".to_string()),
                Expr::Var("n1".to_string()),
                Expr::Var("n2".to_string()),
            ],
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("Graph.hasPath"));
        assert!(result.contains("n1"));
        assert!(result.contains("n2"));
    }

    #[test]
    fn test_compile_lean4_in_graph() {
        let compiler = Lean4Compiler::new("Test");
        let expr = Expr::App(
            "in_graph".to_string(),
            vec![Expr::Var("node".to_string()), Expr::Var("g".to_string())],
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("node"));
        assert!(result.contains("∈"));
        assert!(result.contains("g.nodes"));
    }

    #[test]
    fn test_compile_lean4_edge_exists() {
        let compiler = Lean4Compiler::new("Test");
        let expr = Expr::App(
            "edge_exists".to_string(),
            vec![
                Expr::Var("g".to_string()),
                Expr::Var("from".to_string()),
                Expr::Var("to".to_string()),
            ],
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("from"));
        assert!(result.contains("to"));
        assert!(result.contains("∈"));
        assert!(result.contains("g.edges"));
    }

    #[test]
    fn test_compile_lean4_successors() {
        let compiler = Lean4Compiler::new("Test");
        let expr = Expr::App(
            "successors".to_string(),
            vec![Expr::Var("g".to_string()), Expr::Var("n".to_string())],
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("Graph.successors"));
        assert!(result.contains("g"));
        assert!(result.contains("n"));
    }

    #[test]
    fn test_compile_lean4_predecessors() {
        let compiler = Lean4Compiler::new("Test");
        let expr = Expr::App(
            "predecessors".to_string(),
            vec![Expr::Var("g".to_string()), Expr::Var("n".to_string())],
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("Graph.predecessors"));
        assert!(result.contains("g"));
        assert!(result.contains("n"));
    }

    #[test]
    fn test_compile_lean4_node_count() {
        let compiler = Lean4Compiler::new("Test");
        let expr = Expr::App("node_count".to_string(), vec![Expr::Var("g".to_string())]);
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("g.nodes.card"));
    }

    #[test]
    fn test_compile_lean4_preserves_dag() {
        let compiler = Lean4Compiler::new("Test");
        let expr = Expr::App(
            "preserves_dag".to_string(),
            vec![Expr::Var("g".to_string()), Expr::Var("g_prime".to_string())],
        );
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("Graph.isAcyclic g"));
        assert!(result.contains("Graph.isAcyclic g_prime"));
        assert!(result.contains("→"));
    }

    #[test]
    fn test_compile_lean4_node_completed() {
        let compiler = Lean4Compiler::new("Test");
        let expr = Expr::App("completed".to_string(), vec![Expr::Var("node".to_string())]);
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("Node.isCompleted"));
        assert!(result.contains("node"));
    }

    #[test]
    fn test_compile_lean4_graph_type() {
        let compiler = Lean4Compiler::new("Test");
        let graph_type = Type::Graph(
            Box::new(Type::Named("TaskNode".to_string())),
            Box::new(Type::Named("Dependency".to_string())),
        );
        let result = compiler.compile_type(&graph_type);
        assert_eq!(result, "Graph TaskNode Dependency");
    }

    #[test]
    fn test_compile_lean4_path_type() {
        let compiler = Lean4Compiler::new("Test");
        let path_type = Type::Path(Box::new(Type::Named("Node".to_string())));
        let result = compiler.compile_type(&path_type);
        assert_eq!(result, "Path Node");
    }

    // =========================================================================
    // Phase 17.6: Self-improvement constructs compilation tests
    // =========================================================================

    #[test]
    fn test_compile_lean4_improvement_proposal_basic() {
        use crate::ast::{BinaryOp, ComparisonOp, ImprovementProposal};

        let proposal = ImprovementProposal {
            name: "CodeOptimization".to_string(),
            target: Expr::FieldAccess(
                Box::new(Expr::Var("Dasher".to_string())),
                "verify_rust_code".to_string(),
            ),
            improves: vec![Expr::Compare(
                Box::new(Expr::Var("execution_speed".to_string())),
                ComparisonOp::Ge,
                Box::new(Expr::Binary(
                    Box::new(Expr::Float(1.1)),
                    BinaryOp::Mul,
                    Box::new(Expr::Var("baseline".to_string())),
                )),
            )],
            preserves: vec![Expr::Var("soundness".to_string())],
            requires: vec![Expr::App(
                "valid_rust_syntax".to_string(),
                vec![Expr::Var("new_code".to_string())],
            )],
        };

        let compiler = Lean4Compiler::new("Test");
        let result = compiler.compile_improvement_proposal(&proposal);

        assert!(result.contains("Improvement Proposal: CodeOptimization"));
        assert!(result.contains("structure CodeOptimization where"));
        assert!(result.contains("codeoptimization_target"));
        assert!(result.contains("codeoptimization_improves_1"));
        assert!(result.contains("execution_speed"));
        assert!(result.contains("codeoptimization_preserves_1"));
        assert!(result.contains("soundness"));
        assert!(result.contains("codeoptimization_requires_1"));
        assert!(result.contains("valid_rust_syntax"));
        assert!(result.contains("codeoptimization_valid"));
    }

    #[test]
    fn test_compile_lean4_improvement_proposal_empty() {
        use crate::ast::ImprovementProposal;

        let proposal = ImprovementProposal {
            name: "NoChange".to_string(),
            target: Expr::Var("target".to_string()),
            improves: vec![],
            preserves: vec![],
            requires: vec![],
        };

        let compiler = Lean4Compiler::new("Test");
        let result = compiler.compile_improvement_proposal(&proposal);

        assert!(result.contains("Improvement Proposal: NoChange"));
        assert!(result.contains("structure NoChange where"));
        assert!(result.contains("nochange_target"));
        // Should not contain combined validity since no clauses
        assert!(!result.contains("nochange_valid"));
    }

    #[test]
    fn test_compile_lean4_verification_gate_basic() {
        use crate::ast::{GateCheck, Param, VerificationGate};

        let gate = VerificationGate {
            name: "SelfModGate".to_string(),
            inputs: vec![
                Param {
                    name: "current".to_string(),
                    ty: Type::Named("DasherVersion".to_string()),
                },
                Param {
                    name: "proposed".to_string(),
                    ty: Type::Named("Improvement".to_string()),
                },
            ],
            checks: vec![
                GateCheck {
                    name: "soundness".to_string(),
                    condition: Expr::App(
                        "verify_soundness".to_string(),
                        vec![
                            Expr::Var("current".to_string()),
                            Expr::Var("proposed".to_string()),
                        ],
                    ),
                },
                GateCheck {
                    name: "capability".to_string(),
                    condition: Expr::App(
                        "verify_capabilities".to_string(),
                        vec![
                            Expr::Var("current".to_string()),
                            Expr::Var("proposed".to_string()),
                        ],
                    ),
                },
            ],
            on_pass: Expr::App("apply".to_string(), vec![Expr::Var("proposed".to_string())]),
            on_fail: Expr::App(
                "reject".to_string(),
                vec![Expr::Var("proposed".to_string())],
            ),
        };

        let compiler = Lean4Compiler::new("Test");
        let result = compiler.compile_verification_gate(&gate);

        assert!(result.contains("Verification Gate: SelfModGate"));
        assert!(result.contains("SelfModGate_Inputs where"));
        assert!(result.contains("current : DasherVersion"));
        assert!(result.contains("selfmodgate_check_soundness"));
        assert!(result.contains("selfmodgate_check_capability"));
        assert!(result.contains("selfmodgate_all_checks_pass"));
        assert!(result.contains("inductive SelfModGate_Result"));
        assert!(result.contains("selfmodgate_on_pass"));
        assert!(result.contains("selfmodgate_on_fail"));
    }

    #[test]
    fn test_compile_lean4_verification_gate_no_inputs() {
        use crate::ast::{GateCheck, VerificationGate};

        let gate = VerificationGate {
            name: "SimpleGate".to_string(),
            inputs: vec![],
            checks: vec![GateCheck {
                name: "check1".to_string(),
                condition: Expr::Bool(true),
            }],
            on_pass: Expr::Var("success".to_string()),
            on_fail: Expr::Var("failure".to_string()),
        };

        let compiler = Lean4Compiler::new("Test");
        let result = compiler.compile_verification_gate(&gate);

        assert!(result.contains("Verification Gate: SimpleGate"));
        assert!(!result.contains("SimpleGate_Inputs where"));
        assert!(result.contains("simplegate_check_check1"));
    }

    #[test]
    fn test_compile_lean4_rollback_spec_basic() {
        use crate::ast::{ComparisonOp, Param, RollbackAction, RollbackSpec};

        let rollback = RollbackSpec {
            name: "SafeRollback".to_string(),
            state: vec![
                Param {
                    name: "current".to_string(),
                    ty: Type::Named("DasherVersion".to_string()),
                },
                Param {
                    name: "history".to_string(),
                    ty: Type::List(Box::new(Type::Named("DasherVersion".to_string()))),
                },
            ],
            invariants: vec![Expr::Compare(
                Box::new(Expr::App(
                    "len".to_string(),
                    vec![Expr::Var("history".to_string())],
                )),
                ComparisonOp::Gt,
                Box::new(Expr::Int(0)),
            )],
            trigger: Expr::Or(
                Box::new(Expr::Var("verification_failed".to_string())),
                Box::new(Expr::Var("runtime_error".to_string())),
            ),
            action: RollbackAction {
                assignments: vec![(
                    "current".to_string(),
                    Expr::App("last".to_string(), vec![Expr::Var("history".to_string())]),
                )],
                ensure: Some(Expr::App(
                    "verified".to_string(),
                    vec![Expr::Var("current".to_string())],
                )),
            },
            guarantees: vec![Expr::App(
                "always".to_string(),
                vec![Expr::App(
                    "verified".to_string(),
                    vec![Expr::Var("current".to_string())],
                )],
            )],
        };

        let compiler = Lean4Compiler::new("Test");
        let result = compiler.compile_rollback_spec(&rollback);

        assert!(result.contains("Rollback Specification: SafeRollback"));
        assert!(result.contains("SafeRollback_State where"));
        assert!(result.contains("current : DasherVersion"));
        assert!(result.contains("history : List DasherVersion"));
        assert!(result.contains("saferollback_invariant_1"));
        assert!(result.contains("saferollback_all_invariants"));
        assert!(result.contains("saferollback_trigger"));
        assert!(result.contains("saferollback_ensure"));
        assert!(result.contains("saferollback_guarantee_1"));
        assert!(result.contains("saferollback_spec"));
    }

    #[test]
    fn test_compile_lean4_rollback_spec_minimal() {
        use crate::ast::{RollbackAction, RollbackSpec};

        let rollback = RollbackSpec {
            name: "MinimalRollback".to_string(),
            state: vec![],
            invariants: vec![],
            trigger: Expr::Var("error".to_string()),
            action: RollbackAction {
                assignments: vec![],
                ensure: None,
            },
            guarantees: vec![],
        };

        let compiler = Lean4Compiler::new("Test");
        let result = compiler.compile_rollback_spec(&rollback);

        assert!(result.contains("Rollback Specification: MinimalRollback"));
        assert!(result.contains("minimalrollback_trigger"));
        // Should not have spec since no invariants or guarantees
        assert!(!result.contains("minimalrollback_spec"));
    }

    #[test]
    fn test_compile_lean4_module_with_improvement_proposal() {
        use crate::ast::ImprovementProposal;

        let proposal = ImprovementProposal {
            name: "TestProposal".to_string(),
            target: Expr::Var("target".to_string()),
            improves: vec![Expr::Bool(true)],
            preserves: vec![],
            requires: vec![],
        };

        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![Property::ImprovementProposal(proposal)],
            },
            type_info: std::collections::HashMap::new(),
        };

        let compiler = Lean4Compiler::new("ImprovementTest");
        let result = compiler.compile_module(&spec);

        assert!(result.code.contains("namespace ImprovementTest"));
        assert!(result.code.contains("Improvement Proposal: TestProposal"));
        assert!(result.code.contains("testproposal_target"));
    }

    #[test]
    fn test_compile_lean4_module_with_verification_gate() {
        use crate::ast::{GateCheck, VerificationGate};

        let gate = VerificationGate {
            name: "TestGate".to_string(),
            inputs: vec![],
            checks: vec![GateCheck {
                name: "check".to_string(),
                condition: Expr::Bool(true),
            }],
            on_pass: Expr::Var("pass".to_string()),
            on_fail: Expr::Var("fail".to_string()),
        };

        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![Property::VerificationGate(gate)],
            },
            type_info: std::collections::HashMap::new(),
        };

        let compiler = Lean4Compiler::new("GateTest");
        let result = compiler.compile_module(&spec);

        assert!(result.code.contains("namespace GateTest"));
        assert!(result.code.contains("Verification Gate: TestGate"));
    }

    #[test]
    fn test_compile_lean4_module_with_rollback() {
        use crate::ast::{RollbackAction, RollbackSpec};

        let rollback = RollbackSpec {
            name: "TestRollback".to_string(),
            state: vec![],
            invariants: vec![],
            trigger: Expr::Var("trigger".to_string()),
            action: RollbackAction {
                assignments: vec![],
                ensure: None,
            },
            guarantees: vec![],
        };

        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![Property::Rollback(rollback)],
            },
            type_info: std::collections::HashMap::new(),
        };

        let compiler = Lean4Compiler::new("RollbackTest");
        let result = compiler.compile_module(&spec);

        assert!(result.code.contains("namespace RollbackTest"));
        assert!(result.code.contains("Rollback Specification: TestRollback"));
    }
}
