//! SMT-LIB2 backend compiler
//!
//! Compiles USL specifications to SMT-LIB2 format for Z3, CVC5, and other SMT solvers.

use crate::ast::{
    BinaryOp, ComparisonOp, Contract, Expr, Invariant, Property, Security, Theorem, Type,
};
use crate::typecheck::TypedSpec;

use super::CompiledSpec;

/// SMT-LIB2 compiler for Z3 and CVC5
///
/// Generates SMT-LIB2 standard format that can be consumed by:
/// - Z3 (Microsoft Research)
/// - CVC5 (Stanford/Iowa)
/// - Other SMT-LIB2 compliant solvers
pub struct SmtLib2Compiler {
    /// Logic to use (e.g., "ALL", "QF_LIA", "QF_LRA", "QF_UFLIA")
    logic: String,
}

impl SmtLib2Compiler {
    /// Create a new SMT-LIB2 compiler with default logic (ALL)
    #[must_use]
    pub fn new() -> Self {
        Self {
            logic: "ALL".to_string(),
        }
    }

    /// Create with a specific logic
    #[must_use]
    pub fn with_logic(logic: &str) -> Self {
        Self {
            logic: logic.to_string(),
        }
    }

    /// Compile an expression to SMT-LIB2 syntax
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn compile_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Var(name) => {
                // Convert primed variables to SMT-friendly names
                if name.ends_with('\'') {
                    format!("{}_prime", &name[..name.len() - 1])
                } else {
                    name.clone()
                }
            }
            Expr::Int(n) => {
                if *n < 0 {
                    format!("(- {})", n.abs())
                } else {
                    n.to_string()
                }
            }
            Expr::Float(f) => {
                if *f < 0.0 {
                    format!("(- {})", f.abs())
                } else {
                    f.to_string()
                }
            }
            Expr::String(s) => format!("\"{}\"", s.replace('"', "\"\"")),
            Expr::Bool(b) => if *b { "true" } else { "false" }.to_string(),

            Expr::ForAll { var, ty, body } => {
                let ty_str = ty
                    .as_ref()
                    .map_or_else(|| "Int".to_string(), |t| self.compile_type(t));
                format!(
                    "(forall (({} {})) {})",
                    var,
                    ty_str,
                    self.compile_expr(body)
                )
            }
            Expr::Exists { var, ty, body } => {
                let ty_str = ty
                    .as_ref()
                    .map_or_else(|| "Int".to_string(), |t| self.compile_type(t));
                format!(
                    "(exists (({} {})) {})",
                    var,
                    ty_str,
                    self.compile_expr(body)
                )
            }
            Expr::ForAllIn {
                var,
                collection,
                body,
            } => {
                // In SMT-LIB2, we model "forall x in S" as "forall x . member(x, S) => body"
                format!(
                    "(forall (({} Int)) (=> (member {} {}) {}))",
                    var,
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
                // In SMT-LIB2, we model "exists x in S" as "exists x . member(x, S) and body"
                format!(
                    "(exists (({} Int)) (and (member {} {}) {}))",
                    var,
                    var,
                    self.compile_expr(collection),
                    self.compile_expr(body)
                )
            }

            Expr::Implies(lhs, rhs) => {
                format!("(=> {} {})", self.compile_expr(lhs), self.compile_expr(rhs))
            }
            Expr::And(lhs, rhs) => {
                format!(
                    "(and {} {})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Or(lhs, rhs) => {
                format!("(or {} {})", self.compile_expr(lhs), self.compile_expr(rhs))
            }
            Expr::Not(e) => format!("(not {})", self.compile_expr(e)),

            Expr::Compare(lhs, op, rhs) => {
                let op_str = match op {
                    ComparisonOp::Eq => "=",
                    ComparisonOp::Ne => "distinct",
                    ComparisonOp::Lt => "<",
                    ComparisonOp::Le => "<=",
                    ComparisonOp::Gt => ">",
                    ComparisonOp::Ge => ">=",
                };
                format!(
                    "({} {} {})",
                    op_str,
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Binary(lhs, op, rhs) => {
                let op_str = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "div",
                    BinaryOp::Mod => "mod",
                };
                format!(
                    "({} {} {})",
                    op_str,
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Neg(e) => format!("(- {})", self.compile_expr(e)),

            Expr::App(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let args_str: Vec<String> = args.iter().map(|a| self.compile_expr(a)).collect();
                    format!("({} {})", name, args_str.join(" "))
                }
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                // Convert method calls to function application
                let recv_str = self.compile_expr(receiver);
                if args.is_empty() {
                    format!("({} {})", method, recv_str)
                } else {
                    let args_str: Vec<String> = args.iter().map(|a| self.compile_expr(a)).collect();
                    format!("({} {} {})", method, recv_str, args_str.join(" "))
                }
            }
            Expr::FieldAccess(obj, field) => {
                // Field access as a function application
                format!("({} {})", field, self.compile_expr(obj))
            }
        }
    }

    /// Compile a type to SMT-LIB2 sort
    #[must_use]
    pub fn compile_type(&self, ty: &Type) -> String {
        match ty {
            Type::Named(name) => match name.as_str() {
                "Int" => "Int".to_string(),
                "Float" | "Real" => "Real".to_string(),
                "Bool" => "Bool".to_string(),
                "String" => "String".to_string(),
                _ => name.clone(), // User-defined sorts
            },
            Type::Set(inner) => format!("(Set {})", self.compile_type(inner)),
            Type::List(inner) => format!("(Seq {})", self.compile_type(inner)),
            Type::Map(k, v) => {
                format!("(Array {} {})", self.compile_type(k), self.compile_type(v))
            }
            Type::Relation(a, b) => {
                format!(
                    "(Set (Tuple {} {}))",
                    self.compile_type(a),
                    self.compile_type(b)
                )
            }
            Type::Function(a, b) => {
                format!("(Array {} {})", self.compile_type(a), self.compile_type(b))
            }
            Type::Result(inner) => {
                // Model Result as an optional type
                format!("(Option {})", self.compile_type(inner))
            }
            Type::Unit => "Bool".to_string(), // Unit represented as Bool (true)
            Type::Graph(n, e) => {
                // SMT-LIB graph as pair of node set and edge relation
                format!("(Graph {} {})", self.compile_type(n), self.compile_type(e))
            }
            Type::Path(n) => format!("(Seq {})", self.compile_type(n)),
        }
    }

    /// Collect free variables from an expression
    fn collect_free_vars(
        &self,
        expr: &Expr,
        bound: &mut Vec<String>,
        free: &mut Vec<(String, Option<Type>)>,
    ) {
        match expr {
            Expr::Var(name) => {
                let name_clean = if name.ends_with('\'') {
                    format!("{}_prime", &name[..name.len() - 1])
                } else {
                    name.clone()
                };
                if !bound.contains(&name_clean) && !free.iter().any(|(n, _)| n == &name_clean) {
                    free.push((name_clean, None));
                }
            }
            Expr::ForAll { var, ty, body } | Expr::Exists { var, ty, body } => {
                bound.push(var.clone());
                self.collect_free_vars(body, bound, free);
                bound.pop();
                // Record type hint if available
                if let Some(t) = ty {
                    for (name, existing_ty) in free.iter_mut() {
                        if name == var && existing_ty.is_none() {
                            *existing_ty = Some(t.clone());
                        }
                    }
                }
            }
            Expr::ForAllIn {
                var,
                collection,
                body,
            }
            | Expr::ExistsIn {
                var,
                collection,
                body,
            } => {
                self.collect_free_vars(collection, bound, free);
                bound.push(var.clone());
                self.collect_free_vars(body, bound, free);
                bound.pop();
            }
            Expr::Implies(a, b)
            | Expr::And(a, b)
            | Expr::Or(a, b)
            | Expr::Compare(a, _, b)
            | Expr::Binary(a, _, b) => {
                self.collect_free_vars(a, bound, free);
                self.collect_free_vars(b, bound, free);
            }
            Expr::Not(e) | Expr::Neg(e) => {
                self.collect_free_vars(e, bound, free);
            }
            Expr::App(_, args) => {
                for arg in args {
                    self.collect_free_vars(arg, bound, free);
                }
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.collect_free_vars(receiver, bound, free);
                for arg in args {
                    self.collect_free_vars(arg, bound, free);
                }
            }
            Expr::FieldAccess(obj, _) => {
                self.collect_free_vars(obj, bound, free);
            }
            Expr::Int(_) | Expr::Float(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
    }

    /// Compile a theorem to SMT-LIB2 assertion
    #[must_use]
    pub fn compile_theorem(&self, thm: &Theorem) -> String {
        let body = self.compile_expr(&thm.body);
        format!("; Theorem: {}\n(assert (not {}))\n", thm.name, body)
    }

    /// Compile an invariant to SMT-LIB2 assertion
    #[must_use]
    pub fn compile_invariant(&self, inv: &Invariant) -> String {
        let body = self.compile_expr(&inv.body);
        format!("; Invariant: {}\n(assert (not {}))\n", inv.name, body)
    }

    /// Compile a contract to SMT-LIB2 assertions
    #[must_use]
    pub fn compile_contract(&self, contract: &Contract) -> String {
        let contract_name = contract.type_path.join("::");
        let mut result = format!("; Contract: {}\n", contract_name);

        // Assume preconditions
        for (i, pre) in contract.requires.iter().enumerate() {
            result.push_str(&format!(
                "(assert {}) ; precondition {}\n",
                self.compile_expr(pre),
                i
            ));
        }

        // Check postconditions (negated for satisfiability check)
        for (i, post) in contract.ensures.iter().enumerate() {
            result.push_str(&format!(
                "(assert (not {})) ; postcondition {} (negated)\n",
                self.compile_expr(post),
                i
            ));
        }

        result
    }

    /// Compile a security property to SMT-LIB2
    #[must_use]
    pub fn compile_security(&self, sec: &Security) -> String {
        let body = self.compile_expr(&sec.body);
        format!("; Security: {}\n(assert (not {}))\n", sec.name, body)
    }

    /// Compile a full spec to SMT-LIB2
    #[must_use]
    pub fn compile_module(&self, spec: &TypedSpec) -> CompiledSpec {
        let mut code = String::new();

        // Header
        code.push_str("; Generated by DashProve\n");
        code.push_str(&format!("(set-logic {})\n", self.logic));
        code.push_str("(set-option :produce-models true)\n\n");

        // Declare user-defined sorts from types
        for type_def in &spec.spec.types {
            code.push_str(&format!("; Type: {}\n", type_def.name));
            code.push_str(&format!("(declare-sort {} 0)\n", type_def.name));
        }
        if !spec.spec.types.is_empty() {
            code.push('\n');
        }

        // Collect and declare free variables from all properties
        let mut all_free_vars: Vec<(String, Option<Type>)> = Vec::new();
        for prop in &spec.spec.properties {
            let expr = match prop {
                Property::Theorem(t) => &t.body,
                Property::Invariant(i) => &i.body,
                Property::Security(s) => &s.body,
                Property::Contract(c) => {
                    // Collect from requires and ensures
                    let mut bound = Vec::new();
                    for req in &c.requires {
                        self.collect_free_vars(req, &mut bound, &mut all_free_vars);
                    }
                    for ens in &c.ensures {
                        self.collect_free_vars(ens, &mut bound, &mut all_free_vars);
                    }
                    continue;
                }
                Property::Temporal(_)
                | Property::Refinement(_)
                | Property::Probabilistic(_)
                | Property::Semantic(_)
                | Property::PlatformApi(_)
                | Property::Bisimulation(_)
                | Property::Version(_)
                | Property::Capability(_)
                | Property::DistributedInvariant(_)
                | Property::DistributedTemporal(_)
                | Property::Composed(_)
                | Property::ImprovementProposal(_)
                | Property::VerificationGate(_)
                | Property::Rollback(_) => continue,
            };
            let mut bound = Vec::new();
            self.collect_free_vars(expr, &mut bound, &mut all_free_vars);
        }

        // Declare free variables (constants)
        if !all_free_vars.is_empty() {
            code.push_str("; Free variables\n");
            for (var, ty) in &all_free_vars {
                let sort = ty
                    .as_ref()
                    .map_or_else(|| "Int".to_string(), |t| self.compile_type(t));
                code.push_str(&format!("(declare-const {} {})\n", var, sort));
            }
            code.push('\n');
        }

        // Compile properties
        for prop in &spec.spec.properties {
            match prop {
                Property::Theorem(thm) => {
                    code.push_str(&self.compile_theorem(thm));
                }
                Property::Invariant(inv) => {
                    code.push_str(&self.compile_invariant(inv));
                }
                Property::Contract(contract) => {
                    code.push_str(&self.compile_contract(contract));
                }
                Property::Security(sec) => {
                    code.push_str(&self.compile_security(sec));
                }
                // Temporal, Refinement, Probabilistic, PlatformApi, Bisimulation, Version, Capability, Composed are not directly supported
                Property::Temporal(_)
                | Property::Refinement(_)
                | Property::Probabilistic(_)
                | Property::Semantic(_)
                | Property::PlatformApi(_)
                | Property::Bisimulation(_)
                | Property::Version(_)
                | Property::Capability(_)
                | Property::DistributedInvariant(_)
                | Property::DistributedTemporal(_)
                | Property::Composed(_)
                | Property::ImprovementProposal(_)
                | Property::VerificationGate(_)
                | Property::Rollback(_) => {
                    code.push_str("; Skipped: property type not supported by SMT-LIB2\n");
                }
            }
            code.push('\n');
        }

        // Check satisfiability
        code.push_str("; Check satisfiability (unsat = property holds)\n");
        code.push_str("(check-sat)\n");
        code.push_str("(get-model)\n");

        CompiledSpec {
            backend: "SMT-LIB2".to_string(),
            code,
            module_name: None,
            imports: vec![],
        }
    }
}

impl Default for SmtLib2Compiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Compile to SMT-LIB2 format (for Z3, CVC5, etc.)
#[must_use]
pub fn compile_to_smtlib2(spec: &TypedSpec) -> CompiledSpec {
    let compiler = SmtLib2Compiler::new();
    compiler.compile_module(spec)
}

/// Compile to SMT-LIB2 with a specific logic
#[must_use]
pub fn compile_to_smtlib2_with_logic(spec: &TypedSpec, logic: &str) -> CompiledSpec {
    let compiler = SmtLib2Compiler::with_logic(logic);
    compiler.compile_module(spec)
}

// ========== Kani Proofs ==========

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::ast::{BinaryOp, ComparisonOp, Expr, Type};

    /// Prove that compile_expr never produces empty output for integer literals.
    #[kani::proof]
    fn verify_smtlib2_compile_expr_int_nonempty() {
        let compiler = SmtLib2Compiler::new();
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[kani::proof]
    fn verify_smtlib2_compile_expr_bool_nonempty() {
        let compiler = SmtLib2Compiler::new();
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        assert!(result == "true" || result == "false");
    }

    /// Prove that compile_type always produces non-empty output for named types.
    #[kani::proof]
    fn verify_smtlib2_compile_type_nonempty() {
        let compiler = SmtLib2Compiler::new();
        let ty = Type::Named("T".to_string());
        let result = compiler.compile_type(&ty);
        assert!(!result.is_empty());
    }

    /// Prove that comparison operators compile to valid SMT-LIB2 syntax.
    #[kani::proof]
    fn verify_smtlib2_comparison_valid() {
        let compiler = SmtLib2Compiler::new();
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
        assert!(!result.is_empty());
        // All SMT-LIB2 expressions should start with "("
        assert!(result.starts_with('('));
    }

    /// Prove that binary operators compile to valid SMT-LIB2 syntax.
    #[kani::proof]
    fn verify_smtlib2_binary_ops_nonempty() {
        let compiler = SmtLib2Compiler::new();
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

        let expr = Expr::Binary(
            Box::new(Expr::Var("x".to_string())),
            op,
            Box::new(Expr::Var("y".to_string())),
        );
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        assert!(result.starts_with('('));
    }

    /// Prove that compile_type handles Unit type correctly (maps to Bool).
    #[kani::proof]
    fn verify_smtlib2_compile_type_unit() {
        let compiler = SmtLib2Compiler::new();
        let ty = Type::Unit;
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "Bool");
    }

    /// Prove that Int type maps to Int in SMT-LIB2.
    #[kani::proof]
    fn verify_smtlib2_compile_type_int() {
        let compiler = SmtLib2Compiler::new();
        let ty = Type::Named("Int".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "Int");
    }

    /// Prove that Bool type maps to Bool in SMT-LIB2.
    #[kani::proof]
    fn verify_smtlib2_compile_type_bool() {
        let compiler = SmtLib2Compiler::new();
        let ty = Type::Named("Bool".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "Bool");
    }

    /// Prove that logical And expressions compile correctly.
    #[kani::proof]
    fn verify_smtlib2_and_nonempty() {
        let compiler = SmtLib2Compiler::new();
        let expr = Expr::And(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        assert!(result.starts_with("(and"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Spec, TypeDef};

    fn make_typed_spec(properties: Vec<Property>) -> TypedSpec {
        TypedSpec {
            spec: Spec {
                types: vec![],
                properties,
            },
            type_info: std::collections::HashMap::new(),
        }
    }

    fn make_typed_spec_with_types(properties: Vec<Property>, types: Vec<TypeDef>) -> TypedSpec {
        TypedSpec {
            spec: Spec { types, properties },
            type_info: std::collections::HashMap::new(),
        }
    }

    // ============ compile_expr negative float tests ============

    #[test]
    fn test_compile_expr_negative_float() {
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_expr(&Expr::Float(-5.5));
        assert_eq!(result, "(- 5.5)");
    }

    #[test]
    fn test_compile_expr_positive_float() {
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_expr(&Expr::Float(2.5));
        assert_eq!(result, "2.5");
    }

    #[test]
    fn test_compile_expr_zero_float() {
        let compiler = SmtLib2Compiler::new();
        // 0.0 is not < 0.0, so it should not be negated
        let result = compiler.compile_expr(&Expr::Float(0.0));
        assert_eq!(result, "0");
    }

    // ============ compile_type tests ============

    #[test]
    fn test_compile_type_int() {
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_type(&Type::Named("Int".to_string()));
        assert_eq!(result, "Int");
    }

    #[test]
    fn test_compile_type_float() {
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_type(&Type::Named("Float".to_string()));
        assert_eq!(result, "Real");
    }

    #[test]
    fn test_compile_type_real() {
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_type(&Type::Named("Real".to_string()));
        assert_eq!(result, "Real");
    }

    #[test]
    fn test_compile_type_bool() {
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_type(&Type::Named("Bool".to_string()));
        assert_eq!(result, "Bool");
    }

    #[test]
    fn test_compile_type_string() {
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_type(&Type::Named("String".to_string()));
        assert_eq!(result, "String");
    }

    #[test]
    fn test_compile_type_user_defined() {
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_type(&Type::Named("CustomSort".to_string()));
        assert_eq!(result, "CustomSort");
    }

    // ============ collect_free_vars tests ============

    #[test]
    fn test_collect_free_vars_simple() {
        let compiler = SmtLib2Compiler::new();
        let expr = Expr::Var("x".to_string());
        let mut bound = Vec::new();
        let mut free = Vec::new();
        compiler.collect_free_vars(&expr, &mut bound, &mut free);
        assert_eq!(free.len(), 1);
        assert_eq!(free[0].0, "x");
    }

    #[test]
    fn test_collect_free_vars_bound() {
        let compiler = SmtLib2Compiler::new();
        // ForAll x: body with x used inside should not count x as free
        let expr = Expr::ForAll {
            var: "x".to_string(),
            ty: Some(Type::Named("Int".to_string())),
            body: Box::new(Expr::Var("x".to_string())),
        };
        let mut bound = Vec::new();
        let mut free = Vec::new();
        compiler.collect_free_vars(&expr, &mut bound, &mut free);
        // x is bound, so free should be empty
        assert!(free.is_empty());
    }

    #[test]
    fn test_collect_free_vars_not_duplicate() {
        let compiler = SmtLib2Compiler::new();
        // Same variable used twice should only appear once in free
        let expr = Expr::And(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("x".to_string())),
        );
        let mut bound = Vec::new();
        let mut free = Vec::new();
        compiler.collect_free_vars(&expr, &mut bound, &mut free);
        assert_eq!(free.len(), 1);
    }

    #[test]
    fn test_collect_free_vars_primed() {
        let compiler = SmtLib2Compiler::new();
        let expr = Expr::Var("y'".to_string());
        let mut bound = Vec::new();
        let mut free = Vec::new();
        compiler.collect_free_vars(&expr, &mut bound, &mut free);
        assert_eq!(free.len(), 1);
        assert_eq!(free[0].0, "y_prime");
    }

    #[test]
    fn test_collect_free_vars_forall_in() {
        let compiler = SmtLib2Compiler::new();
        // ForAllIn x in S: x > 0 - x should be bound, S should be free
        let expr = Expr::ForAllIn {
            var: "x".to_string(),
            collection: Box::new(Expr::Var("S".to_string())),
            body: Box::new(Expr::Compare(
                Box::new(Expr::Var("x".to_string())),
                ComparisonOp::Gt,
                Box::new(Expr::Int(0)),
            )),
        };
        let mut bound = Vec::new();
        let mut free = Vec::new();
        compiler.collect_free_vars(&expr, &mut bound, &mut free);
        // S should be free, x should be bound
        assert_eq!(free.len(), 1);
        assert_eq!(free[0].0, "S");
    }

    #[test]
    fn test_collect_free_vars_type_hint_propagation() {
        let compiler = SmtLib2Compiler::new();
        // ForAll with type hint should record the type for variables
        let expr = Expr::And(
            Box::new(Expr::Var("y".to_string())),
            Box::new(Expr::ForAll {
                var: "y".to_string(),
                ty: Some(Type::Named("Int".to_string())),
                body: Box::new(Expr::Bool(true)),
            }),
        );
        let mut bound = Vec::new();
        let mut free = Vec::new();
        compiler.collect_free_vars(&expr, &mut bound, &mut free);
        // y appears free before the forall
        assert!(!free.is_empty());
    }

    // ============ compile_security tests ============

    #[test]
    fn test_compile_security() {
        let compiler = SmtLib2Compiler::new();
        let sec = Security {
            name: "NoOverflow".to_string(),
            body: Expr::Compare(
                Box::new(Expr::Var("x".to_string())),
                ComparisonOp::Le,
                Box::new(Expr::Int(100)),
            ),
        };
        let result = compiler.compile_security(&sec);
        assert!(result.contains("; Security: NoOverflow"));
        assert!(result.contains("(assert (not (<= x 100)))"));
    }

    // ============ compile_module tests ============

    #[test]
    fn test_compile_module_with_types() {
        let spec = make_typed_spec_with_types(
            vec![],
            vec![TypeDef {
                name: "MySort".to_string(),
                fields: vec![],
            }],
        );
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("; Type: MySort"));
        assert!(result.code.contains("(declare-sort MySort 0)"));
        // Should have blank line after types
        assert!(result.code.contains("(declare-sort MySort 0)\n\n"));
    }

    #[test]
    fn test_compile_module_without_types() {
        let spec = make_typed_spec(vec![Property::Theorem(Theorem {
            name: "T".to_string(),
            body: Expr::Bool(true),
        })]);
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_module(&spec);
        // Should NOT have declare-sort
        assert!(!result.code.contains("declare-sort"));
    }

    #[test]
    fn test_compile_module_with_free_vars() {
        let spec = make_typed_spec(vec![Property::Theorem(Theorem {
            name: "HasFreeVar".to_string(),
            body: Expr::Compare(
                Box::new(Expr::Var("x".to_string())),
                ComparisonOp::Eq,
                Box::new(Expr::Int(5)),
            ),
        })]);
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("; Free variables"));
        assert!(result.code.contains("(declare-const x Int)"));
    }

    #[test]
    fn test_compile_module_no_free_vars() {
        let spec = make_typed_spec(vec![Property::Theorem(Theorem {
            name: "NoFreeVars".to_string(),
            body: Expr::Bool(true),
        })]);
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_module(&spec);
        // Should NOT have "Free variables" section
        assert!(!result.code.contains("; Free variables"));
    }

    #[test]
    fn test_compile_module_header() {
        let spec = make_typed_spec(vec![]);
        let compiler = SmtLib2Compiler::with_logic("QF_LIA");
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("; Generated by DashProve"));
        assert!(result.code.contains("(set-logic QF_LIA)"));
        assert!(result.code.contains("(set-option :produce-models true)"));
    }

    #[test]
    fn test_compile_module_footer() {
        let spec = make_typed_spec(vec![]);
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("(check-sat)"));
        assert!(result.code.contains("(get-model)"));
        assert_eq!(result.backend, "SMT-LIB2");
    }

    #[test]
    fn test_compile_module_with_security_property() {
        let spec = make_typed_spec(vec![Property::Security(Security {
            name: "SecProp".to_string(),
            body: Expr::Bool(true),
        })]);
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("; Security: SecProp"));
    }

    // ============ Additional type mapping tests ============

    #[test]
    fn test_compile_type_int_distinct_from_fallback() {
        // Int must map to "Int", not pass through to fallback
        let compiler = SmtLib2Compiler::new();
        // If we deleted the Int match arm, it would return "Int" unchanged
        // (same in this case), but we need another test to distinguish
        // Actually Int -> Int is the same, so the mutation is caught differently
        let result_int = compiler.compile_type(&Type::Named("Int".to_string()));
        let result_real = compiler.compile_type(&Type::Named("Float".to_string()));
        // Int should produce "Int", Float should produce "Real"
        assert_eq!(result_int, "Int");
        assert_eq!(result_real, "Real");
        assert_ne!(result_int, result_real);
    }

    #[test]
    fn test_compile_type_bool_must_be_bool() {
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_type(&Type::Named("Bool".to_string()));
        // Bool must map to "Bool" not fall through
        assert_eq!(result, "Bool");
    }

    #[test]
    fn test_compile_type_string_must_be_string() {
        let compiler = SmtLib2Compiler::new();
        let result = compiler.compile_type(&Type::Named("String".to_string()));
        assert_eq!(result, "String");
    }

    // ============ Type hint propagation tests ============

    #[test]
    fn test_collect_free_vars_type_hint_applied_to_matching_var() {
        let compiler = SmtLib2Compiler::new();
        // Variable y is used free, then bound in a forall with type hint
        // The type hint should propagate to the free variable
        let expr = Expr::And(
            Box::new(Expr::Var("y".to_string())), // y used first (free, no type yet)
            Box::new(Expr::ForAll {
                var: "y".to_string(),
                ty: Some(Type::Named("Int".to_string())),
                body: Box::new(Expr::Bool(true)),
            }),
        );
        let mut bound = Vec::new();
        let mut free = Vec::new();
        compiler.collect_free_vars(&expr, &mut bound, &mut free);

        // y should be recorded as free with type hint Int
        assert_eq!(free.len(), 1);
        assert_eq!(free[0].0, "y");
        assert!(free[0].1.is_some());
        assert_eq!(free[0].1.as_ref().unwrap(), &Type::Named("Int".to_string()));
    }

    #[test]
    fn test_collect_free_vars_type_hint_not_applied_to_wrong_var() {
        let compiler = SmtLib2Compiler::new();
        // Variable x is free, but forall is over y - x should NOT get y's type
        let expr = Expr::And(
            Box::new(Expr::Var("x".to_string())), // x is free
            Box::new(Expr::ForAll {
                var: "y".to_string(), // Different variable
                ty: Some(Type::Named("Bool".to_string())),
                body: Box::new(Expr::Bool(true)),
            }),
        );
        let mut bound = Vec::new();
        let mut free = Vec::new();
        compiler.collect_free_vars(&expr, &mut bound, &mut free);

        // x should be free without type hint (since forall is over y, not x)
        assert_eq!(free.len(), 1);
        assert_eq!(free[0].0, "x");
        assert!(free[0].1.is_none()); // No type hint for x
    }

    #[test]
    fn test_collect_free_vars_type_hint_not_overwrite_existing() {
        let compiler = SmtLib2Compiler::new();
        // Two foralls with different types for the same variable
        // First type hint should win (existing_ty.is_none() check)
        let expr = Expr::And(
            Box::new(Expr::Var("z".to_string())), // z is free
            Box::new(Expr::And(
                Box::new(Expr::ForAll {
                    var: "z".to_string(),
                    ty: Some(Type::Named("Int".to_string())), // First type hint
                    body: Box::new(Expr::Bool(true)),
                }),
                Box::new(Expr::ForAll {
                    var: "z".to_string(),
                    ty: Some(Type::Named("Bool".to_string())), // Second type hint (should be ignored)
                    body: Box::new(Expr::Bool(true)),
                }),
            )),
        );
        let mut bound = Vec::new();
        let mut free = Vec::new();
        compiler.collect_free_vars(&expr, &mut bound, &mut free);

        // z should have first type hint (Int), second should be ignored
        assert_eq!(free.len(), 1);
        assert_eq!(free[0].0, "z");
        assert!(free[0].1.is_some());
        // First type hint wins because existing_ty.is_none() would be false for second
        assert_eq!(free[0].1.as_ref().unwrap(), &Type::Named("Int".to_string()));
    }
}

// =========================================================================
// Kani proofs for SMT-LIB2 compiler correctness
// =========================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::ast::{BinaryOp, ComparisonOp, Expr, Type};

    /// Prove that compile_expr never produces empty output for integer literals.
    #[kani::proof]
    fn verify_smtlib2_compile_expr_int_nonempty() {
        let compiler = SmtLib2Compiler::new();
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[kani::proof]
    fn verify_smtlib2_compile_expr_bool_nonempty() {
        let compiler = SmtLib2Compiler::new();
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // SMT-LIB2 uses lowercase true/false
        assert!(result == "true" || result == "false");
    }

    /// Prove that compile_type always produces non-empty output for named types.
    #[kani::proof]
    fn verify_smtlib2_compile_type_named_nonempty() {
        let compiler = SmtLib2Compiler::new();
        let ty = Type::Named("T".to_string());
        let result = compiler.compile_type(&ty);
        assert!(!result.is_empty());
    }

    /// Prove that compile_type maps Int to Int correctly.
    #[kani::proof]
    fn verify_smtlib2_compile_type_int() {
        let compiler = SmtLib2Compiler::new();
        let ty = Type::Named("Int".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "Int");
    }

    /// Prove that compile_type maps Bool to Bool correctly.
    #[kani::proof]
    fn verify_smtlib2_compile_type_bool() {
        let compiler = SmtLib2Compiler::new();
        let ty = Type::Named("Bool".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "Bool");
    }

    /// Prove that comparison operators compile to valid SMT-LIB2 syntax.
    #[kani::proof]
    fn verify_smtlib2_comparison_valid() {
        let compiler = SmtLib2Compiler::new();
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
        assert!(!result.is_empty());
        // SMT-LIB2 uses prefix notation with parentheses
        assert!(result.starts_with('('));
    }

    /// Prove that binary operators compile to non-empty output with prefix notation.
    #[kani::proof]
    fn verify_smtlib2_binary_ops_nonempty() {
        let compiler = SmtLib2Compiler::new();
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
        // SMT-LIB2 uses prefix notation
        assert!(result.starts_with('('));
    }

    /// Prove that implies compiles to non-empty output with prefix notation.
    #[kani::proof]
    fn verify_smtlib2_implies_nonempty() {
        let compiler = SmtLib2Compiler::new();
        let expr = Expr::Implies(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // SMT-LIB2 uses (=> P Q) for implies
        assert!(result.starts_with("(=>"));
    }

    /// Prove that and compiles to non-empty output with prefix notation.
    #[kani::proof]
    fn verify_smtlib2_and_nonempty() {
        let compiler = SmtLib2Compiler::new();
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // SMT-LIB2 uses (and P Q)
        assert!(result.starts_with("(and"));
    }

    /// Prove that or compiles to non-empty output with prefix notation.
    #[kani::proof]
    fn verify_smtlib2_or_nonempty() {
        let compiler = SmtLib2Compiler::new();
        let expr = Expr::Or(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // SMT-LIB2 uses (or P Q)
        assert!(result.starts_with("(or"));
    }
}
