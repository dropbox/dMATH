//! Dafny backend compiler
//!
//! Compiles USL specifications to Dafny for verification.

use crate::ast::{BinaryOp, ComparisonOp, Contract, Expr, Invariant, Property, Theorem, Type};
use crate::typecheck::TypedSpec;

use super::CompiledSpec;

/// Dafny compiler
pub struct DafnyCompiler {
    module_name: String,
}

impl DafnyCompiler {
    /// Create a new Dafny compiler with the given module name
    #[must_use]
    pub fn new(module_name: &str) -> Self {
        Self {
            module_name: module_name.to_string(),
        }
    }

    /// Compile an expression to Dafny syntax
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn compile_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Var(name) => name.clone(),
            Expr::Int(n) => n.to_string(),
            Expr::Float(f) => f.to_string(),
            Expr::String(s) => format!("\"{s}\""),
            Expr::Bool(b) => if *b { "true" } else { "false" }.to_string(),

            Expr::ForAll { var, ty, body } => {
                let ty_str = ty.as_ref().map_or_else(
                    || ": int".to_string(),
                    |t| format!(": {}", self.compile_type(t)),
                );
                format!("(forall {}{} :: {})", var, ty_str, self.compile_expr(body))
            }
            Expr::Exists { var, ty, body } => {
                let ty_str = ty.as_ref().map_or_else(
                    || ": int".to_string(),
                    |t| format!(": {}", self.compile_type(t)),
                );
                format!("(exists {}{} :: {})", var, ty_str, self.compile_expr(body))
            }
            Expr::ForAllIn {
                var,
                collection,
                body,
            } => {
                format!(
                    "(forall {} :: {} in {} ==> {})",
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
                format!(
                    "(exists {} :: {} in {} && {})",
                    var,
                    var,
                    self.compile_expr(collection),
                    self.compile_expr(body)
                )
            }

            Expr::Implies(lhs, rhs) => {
                format!(
                    "({} ==> {})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::And(lhs, rhs) => {
                format!("({} && {})", self.compile_expr(lhs), self.compile_expr(rhs))
            }
            Expr::Or(lhs, rhs) => {
                format!("({} || {})", self.compile_expr(lhs), self.compile_expr(rhs))
            }
            Expr::Not(e) => format!("(!{})", self.compile_expr(e)),

            Expr::Compare(lhs, op, rhs) => {
                let op_str = match op {
                    ComparisonOp::Eq => "==",
                    ComparisonOp::Ne => "!=",
                    ComparisonOp::Lt => "<",
                    ComparisonOp::Le => "<=",
                    ComparisonOp::Gt => ">",
                    ComparisonOp::Ge => ">=",
                };
                format!(
                    "({} {} {})",
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
                    "({} {} {})",
                    self.compile_expr(lhs),
                    op_str,
                    self.compile_expr(rhs)
                )
            }
            Expr::Neg(e) => format!("(-{})", self.compile_expr(e)),

            Expr::App(name, args) => {
                if args.is_empty() {
                    format!("{name}()")
                } else {
                    let args_str: Vec<_> = args.iter().map(|a| self.compile_expr(a)).collect();
                    format!("{}({})", name, args_str.join(", "))
                }
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                let args_str: Vec<_> = args.iter().map(|a| self.compile_expr(a)).collect();
                if args_str.is_empty() {
                    format!("{}.{}()", self.compile_expr(receiver), method)
                } else {
                    format!(
                        "{}.{}({})",
                        self.compile_expr(receiver),
                        method,
                        args_str.join(", ")
                    )
                }
            }
            Expr::FieldAccess(obj, field) => {
                format!("{}.{}", self.compile_expr(obj), field)
            }
        }
    }

    /// Compile a type to Dafny syntax
    #[must_use]
    pub fn compile_type(&self, ty: &Type) -> String {
        match ty {
            Type::Named(name) => match name.as_str() {
                "Int" => "int".to_string(),
                "Bool" => "bool".to_string(),
                "Nat" => "nat".to_string(),
                "String" => "string".to_string(),
                _ => name.clone(),
            },
            Type::Set(inner) => format!("set<{}>", self.compile_type(inner)),
            Type::List(inner) => format!("seq<{}>", self.compile_type(inner)),
            Type::Map(k, v) => format!("map<{}, {}>", self.compile_type(k), self.compile_type(v)),
            Type::Relation(a, b) => {
                format!("set<({}, {})>", self.compile_type(a), self.compile_type(b))
            }
            Type::Function(a, b) => {
                format!("{} -> {}", self.compile_type(a), self.compile_type(b))
            }
            Type::Result(inner) => format!("Option<{}>", self.compile_type(inner)),
            Type::Unit => "()".to_string(),
            Type::Graph(n, e) => {
                // Dafny graph as datatype with nodes and edges
                format!("Graph<{}, {}>", self.compile_type(n), self.compile_type(e))
            }
            Type::Path(n) => format!("seq<{}>", self.compile_type(n)),
        }
    }

    /// Compile a theorem to Dafny lemma
    #[must_use]
    pub fn compile_theorem(&self, thm: &Theorem) -> String {
        format!(
            "lemma {}\n  ensures {}\n{{\n  // TODO: prove\n}}",
            thm.name,
            self.compile_expr(&thm.body)
        )
    }

    /// Compile an invariant to Dafny lemma
    #[must_use]
    pub fn compile_invariant(&self, inv: &Invariant) -> String {
        format!(
            "lemma {}\n  ensures {}\n{{\n  // TODO: prove\n}}",
            inv.name,
            self.compile_expr(&inv.body)
        )
    }

    /// Compile a contract to Dafny method with pre/post conditions
    #[must_use]
    pub fn compile_contract(&self, contract: &Contract) -> String {
        let name = contract.type_path.join("_");

        let requires = contract
            .requires
            .iter()
            .map(|e| format!("  requires {}", self.compile_expr(e)))
            .collect::<Vec<_>>()
            .join("\n");

        let ensures = contract
            .ensures
            .iter()
            .map(|e| format!("  ensures {}", self.compile_expr(e)))
            .collect::<Vec<_>>()
            .join("\n");

        format!("method {name}\n{requires}\n{ensures}\n{{\n  // TODO: implement\n}}")
    }

    /// Generate complete Dafny module from spec
    #[must_use]
    pub fn compile_module(&self, typed_spec: &TypedSpec) -> CompiledSpec {
        let mut sections = Vec::new();

        // Module header
        sections.push("// Generated by DashProve".to_string());
        sections.push(format!("module {} {{", self.module_name));
        sections.push(String::new());

        // Compile type definitions as datatypes
        for type_def in &typed_spec.spec.types {
            let fields: Vec<_> = type_def
                .fields
                .iter()
                .map(|f| format!("{}: {}", f.name, self.compile_type(&f.ty)))
                .collect();
            sections.push(format!(
                "  datatype {} = {}({})",
                type_def.name,
                type_def.name,
                fields.join(", ")
            ));
            sections.push(String::new());
        }

        // Compile properties
        for property in &typed_spec.spec.properties {
            match property {
                Property::Theorem(thm) => {
                    sections.push(format!("  // Theorem: {}", thm.name));
                    for line in self.compile_theorem(thm).lines() {
                        sections.push(format!("  {line}"));
                    }
                    sections.push(String::new());
                }
                Property::Invariant(inv) => {
                    sections.push(format!("  // Invariant: {}", inv.name));
                    for line in self.compile_invariant(inv).lines() {
                        sections.push(format!("  {line}"));
                    }
                    sections.push(String::new());
                }
                Property::Contract(contract) => {
                    let name = contract.type_path.join("::");
                    sections.push(format!("  // Contract: {name}"));
                    for line in self.compile_contract(contract).lines() {
                        sections.push(format!("  {line}"));
                    }
                    sections.push(String::new());
                }
                _ => {}
            }
        }

        // Module footer
        sections.push("}".to_string());

        CompiledSpec {
            backend: "Dafny".to_string(),
            code: sections.join("\n"),
            module_name: Some(self.module_name.clone()),
            imports: vec![],
        }
    }
}

/// Compile to Dafny
#[must_use]
pub fn compile_to_dafny(spec: &TypedSpec) -> CompiledSpec {
    let compiler = DafnyCompiler::new("USLSpec");
    compiler.compile_module(spec)
}

// ========== Kani Proofs ==========

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::ast::{BinaryOp, ComparisonOp, Expr, Type};

    /// Prove that compile_expr never produces empty output for integer literals.
    #[kani::proof]
    fn verify_dafny_compile_expr_int_nonempty() {
        let compiler = DafnyCompiler::new("Test");
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[kani::proof]
    fn verify_dafny_compile_expr_bool_nonempty() {
        let compiler = DafnyCompiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        assert!(result == "true" || result == "false");
    }

    /// Prove that compile_type always produces non-empty output for named types.
    #[kani::proof]
    fn verify_dafny_compile_type_nonempty() {
        let compiler = DafnyCompiler::new("Test");
        let ty = Type::Named("T".to_string());
        let result = compiler.compile_type(&ty);
        assert!(!result.is_empty());
    }

    /// Prove that comparison operators compile to valid Dafny syntax.
    #[kani::proof]
    fn verify_dafny_comparison_valid() {
        let compiler = DafnyCompiler::new("Test");
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
        // All comparison results should start with "(" in Dafny
        assert!(result.starts_with('('));
    }

    /// Prove that binary operators compile to valid Dafny syntax.
    #[kani::proof]
    fn verify_dafny_binary_ops_nonempty() {
        let compiler = DafnyCompiler::new("Test");
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

    /// Prove that compile_type handles Unit type correctly.
    #[kani::proof]
    fn verify_dafny_compile_type_unit() {
        let compiler = DafnyCompiler::new("Test");
        let ty = Type::Unit;
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "()");
    }

    /// Prove that Int type maps to int in Dafny.
    #[kani::proof]
    fn verify_dafny_compile_type_int() {
        let compiler = DafnyCompiler::new("Test");
        let ty = Type::Named("Int".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "int");
    }

    /// Prove that Bool type maps to bool in Dafny.
    #[kani::proof]
    fn verify_dafny_compile_type_bool() {
        let compiler = DafnyCompiler::new("Test");
        let ty = Type::Named("Bool".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "bool");
    }

    /// Prove that function application always produces non-empty output.
    #[kani::proof]
    fn verify_dafny_app_nonempty() {
        let compiler = DafnyCompiler::new("Test");
        // Test no-arg function call
        let expr = Expr::App("f".to_string(), vec![]);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        assert!(result.ends_with("()"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        BinaryOp, ComparisonOp, Contract, Expr, Field, Invariant, Property, Spec, Theorem, Type,
        TypeDef,
    };
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
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Var("x".to_string())), "x");
    }

    #[test]
    fn test_compile_expr_int() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Int(42)), "42");
        assert_eq!(compiler.compile_expr(&Expr::Int(-5)), "-5");
    }

    #[test]
    fn test_compile_expr_float() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Float(2.71)), "2.71");
    }

    #[test]
    fn test_compile_expr_string() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(
            compiler.compile_expr(&Expr::String("hello".to_string())),
            "\"hello\""
        );
    }

    #[test]
    fn test_compile_expr_bool() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Bool(true)), "true");
        assert_eq!(compiler.compile_expr(&Expr::Bool(false)), "false");
    }

    #[test]
    fn test_compile_expr_forall_with_type() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::ForAll {
            var: "x".to_string(),
            ty: Some(Type::Named("Int".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "(forall x: int :: true)");
    }

    #[test]
    fn test_compile_expr_forall_no_type() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::ForAll {
            var: "x".to_string(),
            ty: None,
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "(forall x: int :: true)");
    }

    #[test]
    fn test_compile_expr_exists_with_type() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::Exists {
            var: "x".to_string(),
            ty: Some(Type::Named("Nat".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "(exists x: nat :: true)");
    }

    #[test]
    fn test_compile_expr_forall_in() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::ForAllIn {
            var: "x".to_string(),
            collection: Box::new(Expr::Var("S".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(
            compiler.compile_expr(&expr),
            "(forall x :: x in S ==> true)"
        );
    }

    #[test]
    fn test_compile_expr_exists_in() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::ExistsIn {
            var: "x".to_string(),
            collection: Box::new(Expr::Var("S".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "(exists x :: x in S && true)");
    }

    #[test]
    fn test_compile_expr_implies() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::Implies(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert_eq!(compiler.compile_expr(&expr), "(a ==> b)");
    }

    #[test]
    fn test_compile_expr_and() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::And(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert_eq!(compiler.compile_expr(&expr), "(a && b)");
    }

    #[test]
    fn test_compile_expr_or() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::Or(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert_eq!(compiler.compile_expr(&expr), "(a || b)");
    }

    #[test]
    fn test_compile_expr_not() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::Not(Box::new(Expr::Var("a".to_string())));
        assert_eq!(compiler.compile_expr(&expr), "(!a)");
    }

    #[test]
    fn test_compile_expr_compare_all_ops() {
        let compiler = DafnyCompiler::new("Test");
        let ops = [
            (ComparisonOp::Eq, "=="),
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
            assert_eq!(compiler.compile_expr(&expr), format!("(x {expected} y)"));
        }
    }

    #[test]
    fn test_compile_expr_binary_all_ops() {
        let compiler = DafnyCompiler::new("Test");
        let ops = [
            (BinaryOp::Add, "+"),
            (BinaryOp::Sub, "-"),
            (BinaryOp::Mul, "*"),
            (BinaryOp::Div, "/"),
            (BinaryOp::Mod, "%"),
        ];
        for (op, expected) in ops {
            let expr = Expr::Binary(
                Box::new(Expr::Var("x".to_string())),
                op,
                Box::new(Expr::Var("y".to_string())),
            );
            assert_eq!(compiler.compile_expr(&expr), format!("(x {expected} y)"));
        }
    }

    #[test]
    fn test_compile_expr_neg() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::Neg(Box::new(Expr::Var("x".to_string())));
        assert_eq!(compiler.compile_expr(&expr), "(-x)");
    }

    #[test]
    fn test_compile_expr_app_no_args() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::App("f".to_string(), vec![]);
        assert_eq!(compiler.compile_expr(&expr), "f()");
    }

    #[test]
    fn test_compile_expr_app_with_args() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::App(
            "f".to_string(),
            vec![Expr::Var("x".to_string()), Expr::Int(1)],
        );
        assert_eq!(compiler.compile_expr(&expr), "f(x, 1)");
    }

    #[test]
    fn test_compile_expr_method_call_no_args() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".to_string())),
            method: "size".to_string(),
            args: vec![],
        };
        assert_eq!(compiler.compile_expr(&expr), "obj.size()");
    }

    #[test]
    fn test_compile_expr_method_call_with_args() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".to_string())),
            method: "get".to_string(),
            args: vec![Expr::Int(1)],
        };
        assert_eq!(compiler.compile_expr(&expr), "obj.get(1)");
    }

    #[test]
    fn test_compile_expr_field_access() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::FieldAccess(Box::new(Expr::Var("obj".to_string())), "field".to_string());
        assert_eq!(compiler.compile_expr(&expr), "obj.field");
    }

    // ========== compile_type tests ==========

    #[test]
    fn test_compile_type_int_maps_to_int() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("Int".to_string())),
            "int"
        );
    }

    #[test]
    fn test_compile_type_bool_maps_to_bool() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("Bool".to_string())),
            "bool"
        );
    }

    #[test]
    fn test_compile_type_nat_maps_to_nat() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("Nat".to_string())),
            "nat"
        );
    }

    #[test]
    fn test_compile_type_string_maps_to_string() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("String".to_string())),
            "string"
        );
    }

    #[test]
    fn test_compile_type_custom_name() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("Person".to_string())),
            "Person"
        );
    }

    #[test]
    fn test_compile_type_set() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Set(Box::new(Type::Named("Int".to_string())))),
            "set<int>"
        );
    }

    #[test]
    fn test_compile_type_list() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::List(Box::new(Type::Named("Int".to_string())))),
            "seq<int>"
        );
    }

    #[test]
    fn test_compile_type_map() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Map(
                Box::new(Type::Named("String".to_string())),
                Box::new(Type::Named("Int".to_string()))
            )),
            "map<string, int>"
        );
    }

    #[test]
    fn test_compile_type_relation() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Relation(
                Box::new(Type::Named("Int".to_string())),
                Box::new(Type::Named("Bool".to_string()))
            )),
            "set<(int, bool)>"
        );
    }

    #[test]
    fn test_compile_type_function() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Function(
                Box::new(Type::Named("Int".to_string())),
                Box::new(Type::Named("Bool".to_string()))
            )),
            "int -> bool"
        );
    }

    #[test]
    fn test_compile_type_result() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Result(Box::new(Type::Named("Int".to_string())))),
            "Option<int>"
        );
    }

    #[test]
    fn test_compile_type_unit() {
        let compiler = DafnyCompiler::new("Test");
        assert_eq!(compiler.compile_type(&Type::Unit), "()");
    }

    // ========== compile_theorem and compile_invariant tests ==========

    #[test]
    fn test_compile_theorem() {
        let compiler = DafnyCompiler::new("Test");
        let thm = Theorem {
            name: "my_theorem".to_string(),
            body: Expr::Bool(true),
        };
        let result = compiler.compile_theorem(&thm);
        assert!(result.contains("lemma my_theorem"));
        assert!(result.contains("ensures true"));
    }

    #[test]
    fn test_compile_invariant() {
        let compiler = DafnyCompiler::new("Test");
        let inv = Invariant {
            name: "my_inv".to_string(),
            body: Expr::Bool(false),
        };
        let result = compiler.compile_invariant(&inv);
        assert!(result.contains("lemma my_inv"));
        assert!(result.contains("ensures false"));
    }

    #[test]
    fn test_compile_contract() {
        let compiler = DafnyCompiler::new("Test");
        let contract = Contract {
            type_path: vec!["MyClass".to_string(), "myMethod".to_string()],
            params: vec![],
            return_type: None,
            requires: vec![Expr::Bool(true)],
            ensures: vec![Expr::Bool(false)],
            ensures_err: vec![],
            assigns: vec![],
            allocates: vec![],
            frees: vec![],
            terminates: None,
            decreases: None,
            behaviors: vec![],
            complete_behaviors: false,
            disjoint_behaviors: false,
        };
        let result = compiler.compile_contract(&contract);
        assert!(result.contains("method MyClass_myMethod"));
        assert!(result.contains("requires true"));
        assert!(result.contains("ensures false"));
    }

    // ========== compile_module tests ==========

    #[test]
    fn test_compile_module_header() {
        let compiler = DafnyCompiler::new("MyModule");
        let spec = make_typed_spec(vec![], vec![]);
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("module MyModule {"));
        assert!(result.code.contains("}"));
        assert_eq!(result.backend, "Dafny");
        assert_eq!(result.module_name, Some("MyModule".to_string()));
    }

    #[test]
    fn test_compile_module_with_type() {
        let compiler = DafnyCompiler::new("Test");
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
        assert!(result
            .code
            .contains("datatype Person = Person(name: string, age: int)"));
    }

    #[test]
    fn test_compile_module_with_theorem() {
        let compiler = DafnyCompiler::new("Test");
        let spec = make_typed_spec(
            vec![],
            vec![Property::Theorem(Theorem {
                name: "thm".to_string(),
                body: Expr::Bool(true),
            })],
        );
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("// Theorem: thm"));
        assert!(result.code.contains("lemma thm"));
    }

    #[test]
    fn test_compile_module_with_invariant() {
        let compiler = DafnyCompiler::new("Test");
        let spec = make_typed_spec(
            vec![],
            vec![Property::Invariant(Invariant {
                name: "inv".to_string(),
                body: Expr::Bool(true),
            })],
        );
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("// Invariant: inv"));
        assert!(result.code.contains("lemma inv"));
    }

    #[test]
    fn test_compile_module_with_contract() {
        let compiler = DafnyCompiler::new("Test");
        let spec = make_typed_spec(
            vec![],
            vec![Property::Contract(Contract {
                type_path: vec!["Foo".to_string(), "bar".to_string()],
                params: vec![],
                return_type: None,
                requires: vec![],
                ensures: vec![],
                ensures_err: vec![],
                assigns: vec![],
                allocates: vec![],
                frees: vec![],
                terminates: None,
                decreases: None,
                behaviors: vec![],
                complete_behaviors: false,
                disjoint_behaviors: false,
            })],
        );
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("// Contract: Foo::bar"));
        assert!(result.code.contains("method Foo_bar"));
    }

    // ========== compile_to_dafny tests ==========

    #[test]
    fn test_compile_to_dafny() {
        let spec = make_typed_spec(vec![], vec![]);
        let result = compile_to_dafny(&spec);
        assert_eq!(result.backend, "Dafny");
        assert_eq!(result.module_name, Some("USLSpec".to_string()));
    }
}

// =========================================================================
// Kani proofs for Dafny compiler correctness
// =========================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::ast::{BinaryOp, ComparisonOp, Expr, Type};

    /// Prove that compile_expr never produces empty output for integer literals.
    #[kani::proof]
    fn verify_dafny_compile_expr_int_nonempty() {
        let compiler = DafnyCompiler::new("Test");
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[kani::proof]
    fn verify_dafny_compile_expr_bool_nonempty() {
        let compiler = DafnyCompiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Dafny uses lowercase true/false
        assert!(result == "true" || result == "false");
    }

    /// Prove that compile_type always produces non-empty output for named types.
    #[kani::proof]
    fn verify_dafny_compile_type_named_nonempty() {
        let compiler = DafnyCompiler::new("Test");
        let ty = Type::Named("T".to_string());
        let result = compiler.compile_type(&ty);
        assert!(!result.is_empty());
    }

    /// Prove that compile_type maps Int to int correctly.
    #[kani::proof]
    fn verify_dafny_compile_type_int() {
        let compiler = DafnyCompiler::new("Test");
        let ty = Type::Named("Int".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "int");
    }

    /// Prove that compile_type maps Bool to bool correctly.
    #[kani::proof]
    fn verify_dafny_compile_type_bool() {
        let compiler = DafnyCompiler::new("Test");
        let ty = Type::Named("Bool".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "bool");
    }

    /// Prove that comparison operators compile to valid Dafny syntax.
    #[kani::proof]
    fn verify_dafny_comparison_valid() {
        let compiler = DafnyCompiler::new("Test");
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
    fn verify_dafny_binary_ops_nonempty() {
        let compiler = DafnyCompiler::new("Test");
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

    /// Prove that implies compiles to non-empty output with ==> operator.
    #[kani::proof]
    fn verify_dafny_implies_nonempty() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::Implies(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Dafny uses ==> for implies
        assert!(result.contains("==>"));
    }

    /// Prove that and compiles to non-empty output.
    #[kani::proof]
    fn verify_dafny_and_nonempty() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Dafny uses && for and
        assert!(result.contains("&&"));
    }

    /// Prove that or compiles to non-empty output.
    #[kani::proof]
    fn verify_dafny_or_nonempty() {
        let compiler = DafnyCompiler::new("Test");
        let expr = Expr::Or(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Dafny uses || for or
        assert!(result.contains("||"));
    }
}
