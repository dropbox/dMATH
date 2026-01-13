//! Alloy backend compiler
//!
//! Compiles USL specifications to Alloy for bounded model checking.

use std::collections::HashSet;

use crate::ast::{BinaryOp, ComparisonOp, Expr, Invariant, Property, Theorem, Type};
use crate::typecheck::TypedSpec;

use super::CompiledSpec;

/// Alloy (bounded model checking) compiler
pub struct AlloyCompiler {
    module_name: String,
}

impl AlloyCompiler {
    /// Create a new Alloy compiler with the given module name
    #[must_use]
    pub fn new(module_name: &str) -> Self {
        Self {
            module_name: module_name.to_string(),
        }
    }

    /// Compile an expression to Alloy syntax
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn compile_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Var(name) => {
                if name.ends_with('\'') {
                    format!("{}'", &name[..name.len() - 1])
                } else {
                    name.clone()
                }
            }
            Expr::Int(n) => n.to_string(),
            Expr::Float(f) => f.to_string(),
            Expr::String(s) => format!("\"{s}\""),
            // Alloy doesn't have standalone boolean literals like "true"/"false"
            // Use idiomatic Alloy: "univ = univ" for true, "no univ" for false
            Expr::Bool(b) => if *b { "univ = univ" } else { "no univ" }.to_string(),

            Expr::ForAll { var, ty, body } => {
                let ty_str = ty
                    .as_ref()
                    .map_or_else(|| "univ".to_string(), |t| self.compile_type(t));
                format!("all {}: {} | {}", var, ty_str, self.compile_expr(body))
            }
            Expr::Exists { var, ty, body } => {
                let ty_str = ty
                    .as_ref()
                    .map_or_else(|| "univ".to_string(), |t| self.compile_type(t));
                format!("some {}: {} | {}", var, ty_str, self.compile_expr(body))
            }
            Expr::ForAllIn {
                var,
                collection,
                body,
            } => {
                format!(
                    "all {}: {} | {}",
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
                    "some {}: {} | {}",
                    var,
                    self.compile_expr(collection),
                    self.compile_expr(body)
                )
            }

            Expr::Implies(lhs, rhs) => {
                format!(
                    "({}) implies ({})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::And(lhs, rhs) => {
                format!(
                    "({}) and ({})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Or(lhs, rhs) => {
                format!(
                    "({}) or ({})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Not(e) => format!("not ({})", self.compile_expr(e)),

            Expr::Compare(lhs, op, rhs) => {
                let op_str = match op {
                    ComparisonOp::Eq => "=",
                    ComparisonOp::Ne => "!=",
                    ComparisonOp::Lt => "<",
                    ComparisonOp::Le => "<=",
                    ComparisonOp::Gt => ">",
                    ComparisonOp::Ge => ">=",
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
                    BinaryOp::Add => ".add",
                    BinaryOp::Sub => ".sub",
                    BinaryOp::Mul => ".mul",
                    BinaryOp::Div => ".div",
                    BinaryOp::Mod => ".rem",
                };
                format!(
                    "({}){}[{}]",
                    self.compile_expr(lhs),
                    op_str,
                    self.compile_expr(rhs)
                )
            }
            Expr::Neg(e) => format!("-({})", self.compile_expr(e)),

            Expr::App(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let args_str: Vec<String> = args.iter().map(|a| self.compile_expr(a)).collect();
                    format!("{}[{}]", name, args_str.join(", "))
                }
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
                    format!("{recv_str}.{method}[{}]", args_str.join(", "))
                }
            }
            Expr::FieldAccess(obj, field) => {
                format!("{}.{field}", self.compile_expr(obj))
            }
        }
    }

    /// Compile a type to Alloy syntax
    #[must_use]
    pub fn compile_type(&self, ty: &Type) -> String {
        match ty {
            Type::Named(name) => name.clone(),
            Type::Set(inner) => format!("set {}", self.compile_type(inner)),
            Type::List(inner) => format!("seq {}", self.compile_type(inner)),
            Type::Map(k, v) => {
                format!("{} -> {}", self.compile_type(k), self.compile_type(v))
            }
            Type::Relation(a, b) => {
                format!("{} -> {}", self.compile_type(a), self.compile_type(b))
            }
            Type::Function(a, b) => {
                format!("{} -> one {}", self.compile_type(a), self.compile_type(b))
            }
            Type::Result(_) => "univ".to_string(),
            Type::Unit => "none".to_string(),
            Type::Graph(n, e) => {
                format!("Graph[{}, {}]", self.compile_type(n), self.compile_type(e))
            }
            Type::Path(n) => format!("seq {}", self.compile_type(n)),
        }
    }

    /// Compile an invariant to Alloy assert (for verification)
    #[must_use]
    pub fn compile_invariant_assert(&self, inv: &Invariant) -> String {
        format!(
            "assert {} {{\n    {}\n}}",
            inv.name,
            self.compile_expr(&inv.body)
        )
    }

    /// Compile a theorem to Alloy assert (for verification)
    #[must_use]
    pub fn compile_theorem_assert(&self, thm: &Theorem) -> String {
        format!(
            "assert {} {{\n    {}\n}}",
            thm.name,
            self.compile_expr(&thm.body)
        )
    }

    /// Collect all function/predicate names referenced in an expression
    /// Returns a set of (name, arity) tuples
    fn collect_referenced_functions(&self, expr: &Expr) -> HashSet<(String, usize)> {
        let mut funcs = HashSet::new();
        self.collect_funcs_recursive(expr, &mut funcs);
        funcs
    }

    fn collect_funcs_recursive(&self, expr: &Expr, funcs: &mut HashSet<(String, usize)>) {
        match expr {
            Expr::App(name, args) => {
                // Collect this function reference with its arity
                funcs.insert((name.clone(), args.len()));
                // Recurse into arguments
                for arg in args {
                    self.collect_funcs_recursive(arg, funcs);
                }
            }
            Expr::ForAll { body, .. } | Expr::Exists { body, .. } => {
                self.collect_funcs_recursive(body, funcs);
            }
            Expr::ForAllIn {
                body, collection, ..
            }
            | Expr::ExistsIn {
                body, collection, ..
            } => {
                self.collect_funcs_recursive(body, funcs);
                self.collect_funcs_recursive(collection, funcs);
            }
            Expr::Implies(lhs, rhs)
            | Expr::And(lhs, rhs)
            | Expr::Or(lhs, rhs)
            | Expr::Compare(lhs, _, rhs)
            | Expr::Binary(lhs, _, rhs) => {
                self.collect_funcs_recursive(lhs, funcs);
                self.collect_funcs_recursive(rhs, funcs);
            }
            Expr::Not(inner) | Expr::Neg(inner) => {
                self.collect_funcs_recursive(inner, funcs);
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.collect_funcs_recursive(receiver, funcs);
                for arg in args {
                    self.collect_funcs_recursive(arg, funcs);
                }
            }
            Expr::FieldAccess(obj, _) => {
                self.collect_funcs_recursive(obj, funcs);
            }
            _ => {} // Var, Int, Float, String, Bool - no functions to collect
        }
    }

    /// Generate pred stub for an undefined function
    fn generate_pred_stub(name: &str, arity: usize) -> String {
        let params: Vec<String> = (0..arity).map(|i| format!("x{i}: univ")).collect();
        format!(
            "// Stub predicate - implement or import actual definition\npred {}[{}] {{}}",
            name,
            params.join(", ")
        )
    }

    /// Generate complete Alloy module from spec
    #[must_use]
    pub fn compile_module(&self, typed_spec: &TypedSpec) -> CompiledSpec {
        let mut sections = Vec::new();
        let mut assertions = Vec::new();
        let mut all_functions = HashSet::new();

        // Module header
        sections.push(format!("module {}", self.module_name));
        sections.push(String::new());

        // Compile type definitions as signatures
        let defined_types: HashSet<_> = typed_spec
            .spec
            .types
            .iter()
            .map(|t| t.name.clone())
            .collect();
        for type_def in &typed_spec.spec.types {
            sections.push(format!("sig {} {{", type_def.name));
            for (i, field) in type_def.fields.iter().enumerate() {
                let comma = if i < type_def.fields.len() - 1 {
                    ","
                } else {
                    ""
                };
                sections.push(format!(
                    "    {}: {}{}",
                    field.name,
                    self.compile_type(&field.ty),
                    comma
                ));
            }
            sections.push("}".to_string());
            sections.push(String::new());
        }

        // Collect all referenced functions from properties
        for property in &typed_spec.spec.properties {
            match property {
                Property::Invariant(inv) => {
                    all_functions.extend(self.collect_referenced_functions(&inv.body));
                }
                Property::Theorem(thm) => {
                    all_functions.extend(self.collect_referenced_functions(&thm.body));
                }
                Property::Security(security) => {
                    all_functions.extend(self.collect_referenced_functions(&security.body));
                }
                _ => {}
            }
        }

        // Generate pred stubs for referenced functions (excluding defined types)
        let mut pred_stubs: Vec<(String, usize)> = all_functions
            .into_iter()
            .filter(|(name, _)| !defined_types.contains(name))
            .collect();
        pred_stubs.sort(); // For deterministic output

        if !pred_stubs.is_empty() {
            sections.push("// Predicate stubs for referenced functions".to_string());
            for (name, arity) in &pred_stubs {
                sections.push(Self::generate_pred_stub(name, *arity));
            }
            sections.push(String::new());
        }

        // Compile properties as assertions
        for property in &typed_spec.spec.properties {
            match property {
                Property::Invariant(inv) => {
                    sections.push(format!("// Invariant: {}", inv.name));
                    sections.push(self.compile_invariant_assert(inv));
                    sections.push(String::new());
                    assertions.push(inv.name.clone());
                }
                Property::Theorem(thm) => {
                    sections.push(format!("// Theorem: {}", thm.name));
                    sections.push(self.compile_theorem_assert(thm));
                    sections.push(String::new());
                    assertions.push(thm.name.clone());
                }
                _ => {}
            }
        }

        // Add check commands for each assertion
        sections.push("// Bounded model checking".to_string());
        if assertions.is_empty() {
            sections.push("run {} for 5".to_string());
        } else {
            for assertion in &assertions {
                sections.push(format!("check {assertion} for 5"));
            }
        }

        CompiledSpec {
            backend: "Alloy".to_string(),
            code: sections.join("\n"),
            module_name: Some(self.module_name.clone()),
            imports: vec![],
        }
    }
}

/// Compile to Alloy
#[must_use]
pub fn compile_to_alloy(spec: &TypedSpec) -> CompiledSpec {
    let compiler = AlloyCompiler::new("USLSpec");
    compiler.compile_module(spec)
}

// ========== Kani Proofs ==========

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::ast::{BinaryOp, ComparisonOp, Expr, Type};

    /// Prove that compile_expr never produces empty output for integer literals.
    #[kani::proof]
    fn verify_alloy_compile_expr_int_nonempty() {
        let compiler = AlloyCompiler::new("Test");
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[kani::proof]
    fn verify_alloy_compile_expr_bool_nonempty() {
        let compiler = AlloyCompiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Alloy uses "univ = univ" for true and "no univ" for false
        assert!(result == "univ = univ" || result == "no univ");
    }

    /// Prove that compile_type always produces non-empty output for named types.
    #[kani::proof]
    fn verify_alloy_compile_type_nonempty() {
        let compiler = AlloyCompiler::new("Test");
        let ty = Type::Named("T".to_string());
        let result = compiler.compile_type(&ty);
        assert!(!result.is_empty());
    }

    /// Prove that comparison operators compile to valid Alloy syntax.
    #[kani::proof]
    fn verify_alloy_comparison_valid() {
        let compiler = AlloyCompiler::new("Test");
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
        // All comparison results should start with "(" in Alloy
        assert!(result.starts_with('('));
    }

    /// Prove that binary operators compile to valid Alloy syntax.
    #[kani::proof]
    fn verify_alloy_binary_ops_nonempty() {
        let compiler = AlloyCompiler::new("Test");
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
        // Alloy binary ops use method-style: (x).add[y]
        assert!(result.starts_with('('));
    }

    /// Prove that compile_type handles Unit type correctly.
    #[kani::proof]
    fn verify_alloy_compile_type_unit() {
        let compiler = AlloyCompiler::new("Test");
        let ty = Type::Unit;
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "none");
    }

    /// Prove that logical operators compile to non-empty output.
    #[kani::proof]
    fn verify_alloy_implies_nonempty() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Implies(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        assert!(result.contains("implies"));
    }

    /// Prove that Not expressions compile correctly.
    #[kani::proof]
    fn verify_alloy_not_nonempty() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Not(Box::new(Expr::Var("a".to_string())));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        assert!(result.starts_with("not"));
    }

    /// Prove that And expressions compile correctly.
    #[kani::proof]
    fn verify_alloy_and_nonempty() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::And(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        assert!(result.contains("and"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        BinaryOp, ComparisonOp, Expr, Field, Invariant, Property, Security, Spec, Theorem, Type,
        TypeDef,
    };
    use crate::typecheck::TypedSpec;
    use std::collections::HashMap;

    fn make_typed_spec(types: Vec<TypeDef>, properties: Vec<Property>) -> TypedSpec {
        TypedSpec {
            spec: Spec { types, properties },
            type_info: HashMap::new(),
        }
    }

    // ========== compile_expr tests ==========

    #[test]
    fn test_compile_expr_var() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Var("x".to_string())), "x");
    }

    #[test]
    fn test_compile_expr_var_primed() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Var("x'".to_string())), "x'");
    }

    #[test]
    fn test_compile_expr_int() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Int(42)), "42");
        assert_eq!(compiler.compile_expr(&Expr::Int(-5)), "-5");
    }

    #[test]
    fn test_compile_expr_float() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Float(2.71)), "2.71");
    }

    #[test]
    fn test_compile_expr_string() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(
            compiler.compile_expr(&Expr::String("hello".to_string())),
            "\"hello\""
        );
    }

    #[test]
    fn test_compile_expr_bool() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Bool(true)), "univ = univ");
        assert_eq!(compiler.compile_expr(&Expr::Bool(false)), "no univ");
    }

    #[test]
    fn test_compile_expr_forall() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::ForAll {
            var: "x".to_string(),
            ty: Some(Type::Named("Int".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "all x: Int | univ = univ");
    }

    #[test]
    fn test_compile_expr_forall_no_type() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::ForAll {
            var: "x".to_string(),
            ty: None,
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "all x: univ | univ = univ");
    }

    #[test]
    fn test_compile_expr_exists() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Exists {
            var: "x".to_string(),
            ty: Some(Type::Named("Int".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "some x: Int | univ = univ");
    }

    #[test]
    fn test_compile_expr_forall_in() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::ForAllIn {
            var: "x".to_string(),
            collection: Box::new(Expr::Var("S".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "all x: S | univ = univ");
    }

    #[test]
    fn test_compile_expr_exists_in() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::ExistsIn {
            var: "x".to_string(),
            collection: Box::new(Expr::Var("S".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "some x: S | univ = univ");
    }

    #[test]
    fn test_compile_expr_implies() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Implies(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert_eq!(compiler.compile_expr(&expr), "(a) implies (b)");
    }

    #[test]
    fn test_compile_expr_and() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::And(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert_eq!(compiler.compile_expr(&expr), "(a) and (b)");
    }

    #[test]
    fn test_compile_expr_or() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Or(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert_eq!(compiler.compile_expr(&expr), "(a) or (b)");
    }

    #[test]
    fn test_compile_expr_not() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Not(Box::new(Expr::Var("a".to_string())));
        assert_eq!(compiler.compile_expr(&expr), "not (a)");
    }

    #[test]
    fn test_compile_expr_compare_all_ops() {
        let compiler = AlloyCompiler::new("Test");
        let ops = [
            (ComparisonOp::Eq, "="),
            (ComparisonOp::Ne, "!="),
            (ComparisonOp::Lt, "<"),
            (ComparisonOp::Le, "<="),
            (ComparisonOp::Gt, ">"),
            (ComparisonOp::Ge, ">="),
        ];
        for (op, expected) in ops {
            let expr = Expr::Compare(
                Box::new(Expr::Var("x".to_string())),
                op,
                Box::new(Expr::Var("y".to_string())),
            );
            assert_eq!(compiler.compile_expr(&expr), format!("(x) {expected} (y)"));
        }
    }

    #[test]
    fn test_compile_expr_binary_all_ops() {
        let compiler = AlloyCompiler::new("Test");
        let ops = [
            (BinaryOp::Add, ".add"),
            (BinaryOp::Sub, ".sub"),
            (BinaryOp::Mul, ".mul"),
            (BinaryOp::Div, ".div"),
            (BinaryOp::Mod, ".rem"),
        ];
        for (op, expected) in ops {
            let expr = Expr::Binary(
                Box::new(Expr::Var("x".to_string())),
                op,
                Box::new(Expr::Var("y".to_string())),
            );
            assert_eq!(compiler.compile_expr(&expr), format!("(x){expected}[y]"));
        }
    }

    #[test]
    fn test_compile_expr_neg() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Neg(Box::new(Expr::Var("x".to_string())));
        assert_eq!(compiler.compile_expr(&expr), "-(x)");
    }

    #[test]
    fn test_compile_expr_app_no_args() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::App("f".to_string(), vec![]);
        assert_eq!(compiler.compile_expr(&expr), "f");
    }

    #[test]
    fn test_compile_expr_app_with_args() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::App(
            "f".to_string(),
            vec![Expr::Var("x".to_string()), Expr::Int(1)],
        );
        assert_eq!(compiler.compile_expr(&expr), "f[x, 1]");
    }

    #[test]
    fn test_compile_expr_method_call_no_args() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".to_string())),
            method: "size".to_string(),
            args: vec![],
        };
        assert_eq!(compiler.compile_expr(&expr), "obj.size");
    }

    #[test]
    fn test_compile_expr_method_call_with_args() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".to_string())),
            method: "get".to_string(),
            args: vec![Expr::Int(1)],
        };
        assert_eq!(compiler.compile_expr(&expr), "obj.get[1]");
    }

    #[test]
    fn test_compile_expr_field_access() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::FieldAccess(Box::new(Expr::Var("obj".to_string())), "field".to_string());
        assert_eq!(compiler.compile_expr(&expr), "obj.field");
    }

    // ========== compile_type tests ==========

    #[test]
    fn test_compile_type_named() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("Person".to_string())),
            "Person"
        );
    }

    #[test]
    fn test_compile_type_set() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Set(Box::new(Type::Named("Int".to_string())))),
            "set Int"
        );
    }

    #[test]
    fn test_compile_type_list() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::List(Box::new(Type::Named("Int".to_string())))),
            "seq Int"
        );
    }

    #[test]
    fn test_compile_type_map() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Map(
                Box::new(Type::Named("String".to_string())),
                Box::new(Type::Named("Int".to_string()))
            )),
            "String -> Int"
        );
    }

    #[test]
    fn test_compile_type_relation() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Relation(
                Box::new(Type::Named("A".to_string())),
                Box::new(Type::Named("B".to_string()))
            )),
            "A -> B"
        );
    }

    #[test]
    fn test_compile_type_function() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Function(
                Box::new(Type::Named("A".to_string())),
                Box::new(Type::Named("B".to_string()))
            )),
            "A -> one B"
        );
    }

    #[test]
    fn test_compile_type_result() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Result(Box::new(Type::Named("Int".to_string())))),
            "univ"
        );
    }

    #[test]
    fn test_compile_type_unit() {
        let compiler = AlloyCompiler::new("Test");
        assert_eq!(compiler.compile_type(&Type::Unit), "none");
    }

    // ========== compile_invariant_assert and compile_theorem_assert tests ==========

    #[test]
    fn test_compile_invariant_assert() {
        let compiler = AlloyCompiler::new("Test");
        let inv = Invariant {
            name: "safety".to_string(),
            body: Expr::Bool(true),
        };
        let result = compiler.compile_invariant_assert(&inv);
        assert!(result.contains("assert safety {"));
        assert!(result.contains("univ = univ"));
    }

    #[test]
    fn test_compile_theorem_assert() {
        let compiler = AlloyCompiler::new("Test");
        let thm = Theorem {
            name: "correctness".to_string(),
            body: Expr::Bool(false),
        };
        let result = compiler.compile_theorem_assert(&thm);
        assert!(result.contains("assert correctness {"));
        assert!(result.contains("no univ"));
    }

    // ========== collect_referenced_functions tests ==========

    #[test]
    fn test_collect_referenced_functions_app() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::App("foo".to_string(), vec![Expr::Int(1), Expr::Int(2)]);
        let funcs = compiler.collect_referenced_functions(&expr);
        assert!(funcs.contains(&("foo".to_string(), 2)));
    }

    #[test]
    fn test_collect_referenced_functions_nested_app() {
        let compiler = AlloyCompiler::new("Test");
        // f(g(x), h(y, z))
        let expr = Expr::App(
            "f".to_string(),
            vec![
                Expr::App("g".to_string(), vec![Expr::Var("x".to_string())]),
                Expr::App(
                    "h".to_string(),
                    vec![Expr::Var("y".to_string()), Expr::Var("z".to_string())],
                ),
            ],
        );
        let funcs = compiler.collect_referenced_functions(&expr);
        assert!(funcs.contains(&("f".to_string(), 2)));
        assert!(funcs.contains(&("g".to_string(), 1)));
        assert!(funcs.contains(&("h".to_string(), 2)));
    }

    #[test]
    fn test_collect_referenced_functions_forall() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::ForAll {
            var: "x".to_string(),
            ty: None,
            body: Box::new(Expr::App(
                "pred".to_string(),
                vec![Expr::Var("x".to_string())],
            )),
        };
        let funcs = compiler.collect_referenced_functions(&expr);
        assert!(funcs.contains(&("pred".to_string(), 1)));
    }

    #[test]
    fn test_collect_referenced_functions_exists() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Exists {
            var: "x".to_string(),
            ty: None,
            body: Box::new(Expr::App(
                "check".to_string(),
                vec![Expr::Var("x".to_string())],
            )),
        };
        let funcs = compiler.collect_referenced_functions(&expr);
        assert!(funcs.contains(&("check".to_string(), 1)));
    }

    #[test]
    fn test_collect_referenced_functions_forall_in() {
        let compiler = AlloyCompiler::new("Test");
        // forall x in getSet(): pred(x)
        let expr = Expr::ForAllIn {
            var: "x".to_string(),
            collection: Box::new(Expr::App("getSet".to_string(), vec![])),
            body: Box::new(Expr::App(
                "pred".to_string(),
                vec![Expr::Var("x".to_string())],
            )),
        };
        let funcs = compiler.collect_referenced_functions(&expr);
        assert!(funcs.contains(&("getSet".to_string(), 0)));
        assert!(funcs.contains(&("pred".to_string(), 1)));
    }

    #[test]
    fn test_collect_referenced_functions_exists_in() {
        let compiler = AlloyCompiler::new("Test");
        // exists x in items(): valid(x)
        let expr = Expr::ExistsIn {
            var: "x".to_string(),
            collection: Box::new(Expr::App("items".to_string(), vec![])),
            body: Box::new(Expr::App(
                "valid".to_string(),
                vec![Expr::Var("x".to_string())],
            )),
        };
        let funcs = compiler.collect_referenced_functions(&expr);
        assert!(funcs.contains(&("items".to_string(), 0)));
        assert!(funcs.contains(&("valid".to_string(), 1)));
    }

    #[test]
    fn test_collect_referenced_functions_implies() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Implies(
            Box::new(Expr::App("p".to_string(), vec![])),
            Box::new(Expr::App("q".to_string(), vec![])),
        );
        let funcs = compiler.collect_referenced_functions(&expr);
        assert!(funcs.contains(&("p".to_string(), 0)));
        assert!(funcs.contains(&("q".to_string(), 0)));
    }

    #[test]
    fn test_collect_referenced_functions_and_or() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::And(
            Box::new(Expr::App("a".to_string(), vec![])),
            Box::new(Expr::Or(
                Box::new(Expr::App("b".to_string(), vec![])),
                Box::new(Expr::App("c".to_string(), vec![])),
            )),
        );
        let funcs = compiler.collect_referenced_functions(&expr);
        assert!(funcs.contains(&("a".to_string(), 0)));
        assert!(funcs.contains(&("b".to_string(), 0)));
        assert!(funcs.contains(&("c".to_string(), 0)));
    }

    #[test]
    fn test_collect_referenced_functions_compare() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Compare(
            Box::new(Expr::App("getX".to_string(), vec![])),
            ComparisonOp::Eq,
            Box::new(Expr::App("getY".to_string(), vec![])),
        );
        let funcs = compiler.collect_referenced_functions(&expr);
        assert!(funcs.contains(&("getX".to_string(), 0)));
        assert!(funcs.contains(&("getY".to_string(), 0)));
    }

    #[test]
    fn test_collect_referenced_functions_binary() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Binary(
            Box::new(Expr::App("left".to_string(), vec![])),
            BinaryOp::Add,
            Box::new(Expr::App("right".to_string(), vec![])),
        );
        let funcs = compiler.collect_referenced_functions(&expr);
        assert!(funcs.contains(&("left".to_string(), 0)));
        assert!(funcs.contains(&("right".to_string(), 0)));
    }

    #[test]
    fn test_collect_referenced_functions_not_neg() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Not(Box::new(Expr::App("notMe".to_string(), vec![])));
        let funcs = compiler.collect_referenced_functions(&expr);
        assert!(funcs.contains(&("notMe".to_string(), 0)));

        let expr2 = Expr::Neg(Box::new(Expr::App("negMe".to_string(), vec![])));
        let funcs2 = compiler.collect_referenced_functions(&expr2);
        assert!(funcs2.contains(&("negMe".to_string(), 0)));
    }

    #[test]
    fn test_collect_referenced_functions_method_call() {
        let compiler = AlloyCompiler::new("Test");
        // receiver.method(arg1())
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::App("getReceiver".to_string(), vec![])),
            method: "call".to_string(),
            args: vec![Expr::App("getArg".to_string(), vec![])],
        };
        let funcs = compiler.collect_referenced_functions(&expr);
        assert!(funcs.contains(&("getReceiver".to_string(), 0)));
        assert!(funcs.contains(&("getArg".to_string(), 0)));
    }

    #[test]
    fn test_collect_referenced_functions_field_access() {
        let compiler = AlloyCompiler::new("Test");
        // getObj().field
        let expr = Expr::FieldAccess(
            Box::new(Expr::App("getObj".to_string(), vec![])),
            "field".to_string(),
        );
        let funcs = compiler.collect_referenced_functions(&expr);
        assert!(funcs.contains(&("getObj".to_string(), 0)));
    }

    #[test]
    fn test_collect_referenced_functions_primitives_empty() {
        let compiler = AlloyCompiler::new("Test");
        assert!(compiler
            .collect_referenced_functions(&Expr::Var("x".to_string()))
            .is_empty());
        assert!(compiler
            .collect_referenced_functions(&Expr::Int(42))
            .is_empty());
        assert!(compiler
            .collect_referenced_functions(&Expr::Float(2.71))
            .is_empty());
        assert!(compiler
            .collect_referenced_functions(&Expr::String("s".to_string()))
            .is_empty());
        assert!(compiler
            .collect_referenced_functions(&Expr::Bool(true))
            .is_empty());
    }

    // ========== generate_pred_stub tests ==========

    #[test]
    fn test_generate_pred_stub_zero_arity() {
        let stub = AlloyCompiler::generate_pred_stub("nullary", 0);
        assert!(stub.contains("pred nullary[]"));
    }

    #[test]
    fn test_generate_pred_stub_one_arity() {
        let stub = AlloyCompiler::generate_pred_stub("unary", 1);
        assert!(stub.contains("pred unary[x0: univ]"));
    }

    #[test]
    fn test_generate_pred_stub_multi_arity() {
        let stub = AlloyCompiler::generate_pred_stub("ternary", 3);
        assert!(stub.contains("pred ternary[x0: univ, x1: univ, x2: univ]"));
    }

    // ========== compile_module tests ==========

    #[test]
    fn test_compile_module_header() {
        let compiler = AlloyCompiler::new("MyModule");
        let spec = make_typed_spec(vec![], vec![]);
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("module MyModule"));
        assert_eq!(result.module_name, Some("MyModule".to_string()));
        assert_eq!(result.backend, "Alloy");
    }

    #[test]
    fn test_compile_module_empty_spec() {
        let compiler = AlloyCompiler::new("Empty");
        let spec = make_typed_spec(vec![], vec![]);
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("run {} for 5"));
    }

    #[test]
    fn test_compile_module_with_type_single_field() {
        let compiler = AlloyCompiler::new("Test");
        let spec = make_typed_spec(
            vec![TypeDef {
                name: "Person".to_string(),
                fields: vec![Field {
                    name: "age".to_string(),
                    ty: Type::Named("Int".to_string()),
                }],
            }],
            vec![],
        );
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("sig Person {"));
        // Single field should NOT have comma
        assert!(result.code.contains("age: Int"));
        assert!(!result.code.contains("age: Int,"));
    }

    #[test]
    fn test_compile_module_with_type_multiple_fields() {
        let compiler = AlloyCompiler::new("Test");
        let spec = make_typed_spec(
            vec![TypeDef {
                name: "Person".to_string(),
                fields: vec![
                    Field {
                        name: "name".to_string(),
                        ty: Type::Named("String".to_string()),
                    },
                    Field {
                        name: "age".to_string(),
                        ty: Type::Named("Int".to_string()),
                    },
                ],
            }],
            vec![],
        );
        let result = compiler.compile_module(&spec);
        // First field should have comma
        assert!(result.code.contains("name: String,"));
        // Last field should NOT have comma
        assert!(result.code.contains("age: Int"));
        // Make sure there's no "age: Int," (comma after last field)
        let lines: Vec<&str> = result.code.lines().collect();
        let age_line = lines.iter().find(|l| l.contains("age: Int")).unwrap();
        assert!(!age_line.ends_with(','));
    }

    #[test]
    fn test_compile_module_with_invariant() {
        let compiler = AlloyCompiler::new("Test");
        let spec = make_typed_spec(
            vec![],
            vec![Property::Invariant(Invariant {
                name: "always_true".to_string(),
                body: Expr::Bool(true),
            })],
        );
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("// Invariant: always_true"));
        assert!(result.code.contains("assert always_true {"));
        assert!(result.code.contains("check always_true for 5"));
    }

    #[test]
    fn test_compile_module_with_theorem() {
        let compiler = AlloyCompiler::new("Test");
        let spec = make_typed_spec(
            vec![],
            vec![Property::Theorem(Theorem {
                name: "my_theorem".to_string(),
                body: Expr::Bool(false),
            })],
        );
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("// Theorem: my_theorem"));
        assert!(result.code.contains("assert my_theorem {"));
        assert!(result.code.contains("check my_theorem for 5"));
    }

    #[test]
    fn test_compile_module_with_security() {
        let compiler = AlloyCompiler::new("Test");
        let spec = make_typed_spec(
            vec![],
            vec![Property::Security(Security {
                name: "sec_prop".to_string(),
                body: Expr::App("secure".to_string(), vec![]),
            })],
        );
        let result = compiler.compile_module(&spec);
        // Security properties should contribute to function collection
        // but not generate an assertion in the module
        assert!(result.code.contains("pred secure[]"));
    }

    #[test]
    fn test_compile_module_pred_stubs_for_invariant_functions() {
        let compiler = AlloyCompiler::new("Test");
        let spec = make_typed_spec(
            vec![],
            vec![Property::Invariant(Invariant {
                name: "inv".to_string(),
                body: Expr::App("myPred".to_string(), vec![Expr::Var("x".to_string())]),
            })],
        );
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("pred myPred[x0: univ]"));
    }

    #[test]
    fn test_compile_module_pred_stubs_for_theorem_functions() {
        let compiler = AlloyCompiler::new("Test");
        let spec = make_typed_spec(
            vec![],
            vec![Property::Theorem(Theorem {
                name: "thm".to_string(),
                body: Expr::App("thmPred".to_string(), vec![]),
            })],
        );
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("pred thmPred[]"));
    }

    #[test]
    fn test_compile_module_excludes_defined_types_from_pred_stubs() {
        let compiler = AlloyCompiler::new("Test");
        let spec = make_typed_spec(
            vec![TypeDef {
                name: "Person".to_string(),
                fields: vec![],
            }],
            vec![Property::Invariant(Invariant {
                name: "inv".to_string(),
                // "Person" is a defined type, so it shouldn't generate a pred stub
                body: Expr::App("Person".to_string(), vec![]),
            })],
        );
        let result = compiler.compile_module(&spec);
        assert!(!result.code.contains("pred Person[]"));
    }

    #[test]
    fn test_compile_module_multiple_assertions() {
        let compiler = AlloyCompiler::new("Test");
        let spec = make_typed_spec(
            vec![],
            vec![
                Property::Invariant(Invariant {
                    name: "inv1".to_string(),
                    body: Expr::Bool(true),
                }),
                Property::Theorem(Theorem {
                    name: "thm1".to_string(),
                    body: Expr::Bool(true),
                }),
            ],
        );
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("check inv1 for 5"));
        assert!(result.code.contains("check thm1 for 5"));
    }

    // ========== compile_to_alloy tests ==========

    #[test]
    fn test_compile_to_alloy() {
        let spec = make_typed_spec(vec![], vec![]);
        let result = compile_to_alloy(&spec);
        assert_eq!(result.backend, "Alloy");
        assert_eq!(result.module_name, Some("USLSpec".to_string()));
    }
}

// =========================================================================
// Kani proofs for Alloy compiler correctness
// =========================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::ast::{BinaryOp, ComparisonOp, Expr, Type};

    /// Prove that compile_expr never produces empty output for integer literals.
    #[kani::proof]
    fn verify_alloy_compile_expr_int_nonempty() {
        let compiler = AlloyCompiler::new("Test");
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[kani::proof]
    fn verify_alloy_compile_expr_bool_nonempty() {
        let compiler = AlloyCompiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_type always produces non-empty output for named types.
    #[kani::proof]
    fn verify_alloy_compile_type_named_nonempty() {
        let compiler = AlloyCompiler::new("Test");
        let ty = Type::Named("T".to_string());
        let result = compiler.compile_type(&ty);
        assert!(!result.is_empty());
    }

    /// Prove that compile_type handles Unit type correctly.
    #[kani::proof]
    fn verify_alloy_compile_type_unit() {
        let compiler = AlloyCompiler::new("Test");
        let ty = Type::Unit;
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "none");
    }

    /// Prove that comparison operators compile to valid Alloy syntax.
    #[kani::proof]
    fn verify_alloy_comparison_valid() {
        let compiler = AlloyCompiler::new("Test");
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
    }

    /// Prove that binary operators compile to non-empty output.
    #[kani::proof]
    fn verify_alloy_binary_ops_nonempty() {
        let compiler = AlloyCompiler::new("Test");
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

    /// Prove that implies compiles to non-empty output.
    #[kani::proof]
    fn verify_alloy_implies_nonempty() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Implies(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Alloy implies uses =>
        assert!(result.contains("=>"));
    }

    /// Prove that and compiles to non-empty output.
    #[kani::proof]
    fn verify_alloy_and_nonempty() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Alloy and uses &&
        assert!(result.contains("&&") || result.contains("and"));
    }

    /// Prove that or compiles to non-empty output.
    #[kani::proof]
    fn verify_alloy_or_nonempty() {
        let compiler = AlloyCompiler::new("Test");
        let expr = Expr::Or(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Alloy or uses ||
        assert!(result.contains("||") || result.contains("or"));
    }

    /// Prove that not compiles to non-empty output.
    #[kani::proof]
    fn verify_alloy_not_nonempty() {
        let compiler = AlloyCompiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Not(Box::new(Expr::Bool(b)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Alloy not uses !
        assert!(result.contains('!') || result.contains("not"));
    }
}
