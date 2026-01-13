//! Coq backend compiler
//!
//! Compiles USL specifications to Coq for theorem proving.

use crate::ast::{BinaryOp, ComparisonOp, Expr, Invariant, Property, Theorem, Type};
use crate::typecheck::TypedSpec;

use super::CompiledSpec;

/// Coq compiler
pub struct CoqCompiler {
    module_name: String,
}

impl CoqCompiler {
    /// Create a new Coq compiler with the given module name
    #[must_use]
    pub fn new(module_name: &str) -> Self {
        Self {
            module_name: module_name.to_string(),
        }
    }

    /// Compile an expression to Coq syntax
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn compile_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Var(name) => name.clone(),
            Expr::Int(n) => {
                if *n < 0 {
                    format!("({n})")
                } else {
                    n.to_string()
                }
            }
            Expr::Float(f) => f.to_string(),
            Expr::String(s) => format!("\"{s}\""),
            Expr::Bool(b) => if *b { "true" } else { "false" }.to_string(),

            Expr::ForAll { var, ty, body } => {
                let ty_str = ty
                    .as_ref()
                    .map(|t| format!(": {}", self.compile_type(t)))
                    .unwrap_or_default();
                format!("(forall {}{}, {})", var, ty_str, self.compile_expr(body))
            }
            Expr::Exists { var, ty, body } => {
                let ty_str = ty
                    .as_ref()
                    .map(|t| format!(": {}", self.compile_type(t)))
                    .unwrap_or_default();
                format!("(exists {}{}, {})", var, ty_str, self.compile_expr(body))
            }
            Expr::ForAllIn {
                var,
                collection,
                body,
            } => {
                format!(
                    "(forall {}, In {} {} -> {})",
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
                    "(exists {}, In {} {} /\\ {})",
                    var,
                    var,
                    self.compile_expr(collection),
                    self.compile_expr(body)
                )
            }

            Expr::Implies(lhs, rhs) => {
                format!("({} -> {})", self.compile_expr(lhs), self.compile_expr(rhs))
            }
            Expr::And(lhs, rhs) => {
                format!(
                    "({} /\\ {})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Or(lhs, rhs) => {
                format!(
                    "({} \\/ {})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Not(e) => format!("(~ {})", self.compile_expr(e)),

            Expr::Compare(lhs, op, rhs) => {
                let op_str = match op {
                    ComparisonOp::Eq => "=",
                    ComparisonOp::Ne => "<>",
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
                    BinaryOp::Mod => "mod",
                };
                format!(
                    "({} {} {})",
                    self.compile_expr(lhs),
                    op_str,
                    self.compile_expr(rhs)
                )
            }
            Expr::Neg(e) => format!("(- {})", self.compile_expr(e)),

            Expr::App(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let args_str: Vec<_> = args.iter().map(|a| self.compile_expr(a)).collect();
                    format!("({} {})", name, args_str.join(" "))
                }
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                let args_str: Vec<_> = args.iter().map(|a| self.compile_expr(a)).collect();
                if args_str.is_empty() {
                    format!("({} {})", method, self.compile_expr(receiver))
                } else {
                    format!(
                        "({} {} {})",
                        method,
                        self.compile_expr(receiver),
                        args_str.join(" ")
                    )
                }
            }
            Expr::FieldAccess(obj, field) => {
                format!("({} {})", field, self.compile_expr(obj))
            }
        }
    }

    /// Compile a type to Coq syntax
    #[must_use]
    pub fn compile_type(&self, ty: &Type) -> String {
        match ty {
            Type::Named(name) => match name.as_str() {
                "Int" => "Z".to_string(),
                "Bool" => "bool".to_string(),
                "Nat" => "nat".to_string(),
                "String" => "string".to_string(),
                _ => name.clone(),
            },
            Type::Set(inner) | Type::List(inner) => format!("list {}", self.compile_type(inner)),
            Type::Map(k, v) => format!("{} -> {}", self.compile_type(k), self.compile_type(v)),
            Type::Relation(a, b) => {
                format!("list ({} * {})", self.compile_type(a), self.compile_type(b))
            }
            Type::Function(a, b) => {
                format!("{} -> {}", self.compile_type(a), self.compile_type(b))
            }
            Type::Result(inner) => format!("option {}", self.compile_type(inner)),
            Type::Unit => "unit".to_string(),
            Type::Graph(n, e) => {
                // Coq graph as record with nodes list and edges relation
                format!(
                    "{{ nodes : list {}; edges : list ({} * {} * {}) }}",
                    self.compile_type(n),
                    self.compile_type(n),
                    self.compile_type(e),
                    self.compile_type(n)
                )
            }
            Type::Path(n) => format!("list {}", self.compile_type(n)),
        }
    }

    /// Compile a theorem to Coq Theorem
    #[must_use]
    pub fn compile_theorem(&self, thm: &Theorem) -> String {
        format!(
            "Theorem {} : {}.\nProof.\n  (* TODO: prove *)\nAdmitted.",
            thm.name,
            self.compile_expr(&thm.body)
        )
    }

    /// Compile an invariant to Coq Lemma
    #[must_use]
    pub fn compile_invariant(&self, inv: &Invariant) -> String {
        format!(
            "Lemma {} : {}.\nProof.\n  (* TODO: prove *)\nAdmitted.",
            inv.name,
            self.compile_expr(&inv.body)
        )
    }

    /// Generate complete Coq module from spec
    #[must_use]
    pub fn compile_module(&self, typed_spec: &TypedSpec) -> CompiledSpec {
        let mut sections = Vec::new();

        // Imports
        sections.push("(* Generated by DashProve *)".to_string());
        sections.push("Require Import Coq.ZArith.ZArith.".to_string());
        sections.push("Require Import Coq.Lists.List.".to_string());
        sections.push("Require Import Coq.Strings.String.".to_string());
        sections.push("Import ListNotations.".to_string());
        sections.push("Open Scope Z_scope.".to_string());
        sections.push(String::new());

        // Module
        sections.push(format!("Module {}.", self.module_name));
        sections.push(String::new());

        // Compile type definitions as Records
        for type_def in &typed_spec.spec.types {
            sections.push(format!("Record {} := {{", type_def.name));
            for (i, field) in type_def.fields.iter().enumerate() {
                let sep = if i < type_def.fields.len() - 1 {
                    ";"
                } else {
                    ""
                };
                sections.push(format!(
                    "  {} : {}{}",
                    field.name,
                    self.compile_type(&field.ty),
                    sep
                ));
            }
            sections.push("}.".to_string());
            sections.push(String::new());
        }

        // Compile properties
        for property in &typed_spec.spec.properties {
            match property {
                Property::Theorem(thm) => {
                    sections.push(format!("(* Theorem: {} *)", thm.name));
                    sections.push(self.compile_theorem(thm));
                    sections.push(String::new());
                }
                Property::Invariant(inv) => {
                    sections.push(format!("(* Invariant: {} *)", inv.name));
                    sections.push(self.compile_invariant(inv));
                    sections.push(String::new());
                }
                _ => {}
            }
        }

        // Module footer
        sections.push(format!("End {}.", self.module_name));

        CompiledSpec {
            backend: "Coq".to_string(),
            code: sections.join("\n"),
            module_name: Some(self.module_name.clone()),
            imports: vec![
                "Coq.ZArith.ZArith".to_string(),
                "Coq.Lists.List".to_string(),
            ],
        }
    }
}

/// Compile to Coq
#[must_use]
pub fn compile_to_coq(spec: &TypedSpec) -> CompiledSpec {
    let compiler = CoqCompiler::new("USLSpec");
    compiler.compile_module(spec)
}

// ========== Kani Proofs ==========

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::ast::{BinaryOp, ComparisonOp, Expr, Type};

    /// Prove that compile_expr never produces empty output for integer literals.
    #[kani::proof]
    fn verify_coq_compile_expr_int_nonempty() {
        let compiler = CoqCompiler::new("Test");
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[kani::proof]
    fn verify_coq_compile_expr_bool_nonempty() {
        let compiler = CoqCompiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        assert!(result == "true" || result == "false");
    }

    /// Prove that compile_type always produces non-empty output for named types.
    #[kani::proof]
    fn verify_coq_compile_type_nonempty() {
        let compiler = CoqCompiler::new("Test");
        let ty = Type::Named("T".to_string());
        let result = compiler.compile_type(&ty);
        assert!(!result.is_empty());
    }

    /// Prove that comparison operators compile to valid Coq syntax.
    #[kani::proof]
    fn verify_coq_comparison_valid() {
        let compiler = CoqCompiler::new("Test");
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
        // All comparison results should start with "(" in Coq
        assert!(result.starts_with('('));
    }

    /// Prove that binary operators compile to valid Coq syntax.
    #[kani::proof]
    fn verify_coq_binary_ops_nonempty() {
        let compiler = CoqCompiler::new("Test");
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
    fn verify_coq_compile_type_unit() {
        let compiler = CoqCompiler::new("Test");
        let ty = Type::Unit;
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "unit");
    }

    /// Prove that Int type maps to Z in Coq.
    #[kani::proof]
    fn verify_coq_compile_type_int_to_z() {
        let compiler = CoqCompiler::new("Test");
        let ty = Type::Named("Int".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "Z");
    }

    /// Prove that Bool type maps to bool in Coq.
    #[kani::proof]
    fn verify_coq_compile_type_bool() {
        let compiler = CoqCompiler::new("Test");
        let ty = Type::Named("Bool".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "bool");
    }

    /// Prove that negative integers are wrapped in parentheses.
    #[kani::proof]
    fn verify_coq_negative_int_parenthesized() {
        let compiler = CoqCompiler::new("Test");
        let n: i64 = kani::any();
        kani::assume(n < 0);
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(result.starts_with('('));
        assert!(result.ends_with(')'));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        BinaryOp, ComparisonOp, Expr, Field, Invariant, Property, Spec, Theorem, Type, TypeDef,
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
        let compiler = CoqCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Var("x".to_string())), "x");
    }

    #[test]
    fn test_compile_expr_int_positive() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Int(42)), "42");
        assert_eq!(compiler.compile_expr(&Expr::Int(0)), "0");
    }

    #[test]
    fn test_compile_expr_int_negative() {
        let compiler = CoqCompiler::new("Test");
        // Negative numbers get wrapped in parentheses
        assert_eq!(compiler.compile_expr(&Expr::Int(-5)), "(-5)");
        assert_eq!(compiler.compile_expr(&Expr::Int(-1)), "(-1)");
    }

    #[test]
    fn test_compile_expr_float() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Float(2.71)), "2.71");
    }

    #[test]
    fn test_compile_expr_string() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(
            compiler.compile_expr(&Expr::String("hello".to_string())),
            "\"hello\""
        );
    }

    #[test]
    fn test_compile_expr_bool() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Bool(true)), "true");
        assert_eq!(compiler.compile_expr(&Expr::Bool(false)), "false");
    }

    #[test]
    fn test_compile_expr_forall_with_type() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::ForAll {
            var: "x".to_string(),
            ty: Some(Type::Named("Int".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "(forall x: Z, true)");
    }

    #[test]
    fn test_compile_expr_forall_no_type() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::ForAll {
            var: "x".to_string(),
            ty: None,
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "(forall x, true)");
    }

    #[test]
    fn test_compile_expr_exists_with_type() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::Exists {
            var: "x".to_string(),
            ty: Some(Type::Named("Nat".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "(exists x: nat, true)");
    }

    #[test]
    fn test_compile_expr_forall_in() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::ForAllIn {
            var: "x".to_string(),
            collection: Box::new(Expr::Var("S".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "(forall x, In x S -> true)");
    }

    #[test]
    fn test_compile_expr_exists_in() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::ExistsIn {
            var: "x".to_string(),
            collection: Box::new(Expr::Var("S".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert_eq!(compiler.compile_expr(&expr), "(exists x, In x S /\\ true)");
    }

    #[test]
    fn test_compile_expr_implies() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::Implies(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert_eq!(compiler.compile_expr(&expr), "(a -> b)");
    }

    #[test]
    fn test_compile_expr_and() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::And(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert_eq!(compiler.compile_expr(&expr), "(a /\\ b)");
    }

    #[test]
    fn test_compile_expr_or() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::Or(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert_eq!(compiler.compile_expr(&expr), "(a \\/ b)");
    }

    #[test]
    fn test_compile_expr_not() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::Not(Box::new(Expr::Var("a".to_string())));
        assert_eq!(compiler.compile_expr(&expr), "(~ a)");
    }

    #[test]
    fn test_compile_expr_compare_all_ops() {
        let compiler = CoqCompiler::new("Test");
        let ops = [
            (ComparisonOp::Eq, "="),
            (ComparisonOp::Ne, "<>"),
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
        let compiler = CoqCompiler::new("Test");
        let ops = [
            (BinaryOp::Add, "+"),
            (BinaryOp::Sub, "-"),
            (BinaryOp::Mul, "*"),
            (BinaryOp::Div, "/"),
            (BinaryOp::Mod, "mod"),
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
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::Neg(Box::new(Expr::Var("x".to_string())));
        assert_eq!(compiler.compile_expr(&expr), "(- x)");
    }

    #[test]
    fn test_compile_expr_app_no_args() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::App("f".to_string(), vec![]);
        assert_eq!(compiler.compile_expr(&expr), "f");
    }

    #[test]
    fn test_compile_expr_app_with_args() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::App(
            "f".to_string(),
            vec![Expr::Var("x".to_string()), Expr::Int(1)],
        );
        assert_eq!(compiler.compile_expr(&expr), "(f x 1)");
    }

    #[test]
    fn test_compile_expr_method_call_no_args() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".to_string())),
            method: "size".to_string(),
            args: vec![],
        };
        assert_eq!(compiler.compile_expr(&expr), "(size obj)");
    }

    #[test]
    fn test_compile_expr_method_call_with_args() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".to_string())),
            method: "get".to_string(),
            args: vec![Expr::Int(1)],
        };
        assert_eq!(compiler.compile_expr(&expr), "(get obj 1)");
    }

    #[test]
    fn test_compile_expr_field_access() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::FieldAccess(Box::new(Expr::Var("obj".to_string())), "field".to_string());
        assert_eq!(compiler.compile_expr(&expr), "(field obj)");
    }

    // ========== compile_type tests ==========

    #[test]
    fn test_compile_type_int_maps_to_z() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(compiler.compile_type(&Type::Named("Int".to_string())), "Z");
    }

    #[test]
    fn test_compile_type_bool_maps_to_bool() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("Bool".to_string())),
            "bool"
        );
    }

    #[test]
    fn test_compile_type_nat_maps_to_nat() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("Nat".to_string())),
            "nat"
        );
    }

    #[test]
    fn test_compile_type_string_maps_to_string() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("String".to_string())),
            "string"
        );
    }

    #[test]
    fn test_compile_type_custom_name() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("Person".to_string())),
            "Person"
        );
    }

    #[test]
    fn test_compile_type_set() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Set(Box::new(Type::Named("Int".to_string())))),
            "list Z"
        );
    }

    #[test]
    fn test_compile_type_list() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::List(Box::new(Type::Named("Int".to_string())))),
            "list Z"
        );
    }

    #[test]
    fn test_compile_type_map() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Map(
                Box::new(Type::Named("String".to_string())),
                Box::new(Type::Named("Int".to_string()))
            )),
            "string -> Z"
        );
    }

    #[test]
    fn test_compile_type_relation() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Relation(
                Box::new(Type::Named("Int".to_string())),
                Box::new(Type::Named("Bool".to_string()))
            )),
            "list (Z * bool)"
        );
    }

    #[test]
    fn test_compile_type_function() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Function(
                Box::new(Type::Named("Int".to_string())),
                Box::new(Type::Named("Bool".to_string()))
            )),
            "Z -> bool"
        );
    }

    #[test]
    fn test_compile_type_result() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Result(Box::new(Type::Named("Int".to_string())))),
            "option Z"
        );
    }

    #[test]
    fn test_compile_type_unit() {
        let compiler = CoqCompiler::new("Test");
        assert_eq!(compiler.compile_type(&Type::Unit), "unit");
    }

    // ========== compile_theorem and compile_invariant tests ==========

    #[test]
    fn test_compile_theorem() {
        let compiler = CoqCompiler::new("Test");
        let thm = Theorem {
            name: "my_theorem".to_string(),
            body: Expr::Bool(true),
        };
        let result = compiler.compile_theorem(&thm);
        assert!(result.contains("Theorem my_theorem : true."));
        assert!(result.contains("Proof."));
        assert!(result.contains("Admitted."));
    }

    #[test]
    fn test_compile_invariant() {
        let compiler = CoqCompiler::new("Test");
        let inv = Invariant {
            name: "my_inv".to_string(),
            body: Expr::Bool(false),
        };
        let result = compiler.compile_invariant(&inv);
        assert!(result.contains("Lemma my_inv : false."));
        assert!(result.contains("Proof."));
        assert!(result.contains("Admitted."));
    }

    // ========== compile_module tests ==========

    #[test]
    fn test_compile_module_header() {
        let compiler = CoqCompiler::new("MyModule");
        let spec = make_typed_spec(vec![], vec![]);
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("Module MyModule."));
        assert!(result.code.contains("End MyModule."));
        assert!(result.code.contains("Require Import Coq.ZArith.ZArith."));
        assert_eq!(result.backend, "Coq");
        assert_eq!(result.module_name, Some("MyModule".to_string()));
    }

    #[test]
    fn test_compile_module_with_type_single_field() {
        let compiler = CoqCompiler::new("Test");
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
        assert!(result.code.contains("Record Person := {"));
        // Single field should NOT have semicolon
        assert!(result.code.contains("age : Z"));
        let lines: Vec<&str> = result.code.lines().collect();
        let age_line = lines.iter().find(|l| l.contains("age : Z")).unwrap();
        assert!(!age_line.ends_with(';'));
    }

    #[test]
    fn test_compile_module_with_type_multiple_fields() {
        let compiler = CoqCompiler::new("Test");
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
        // First field should have semicolon
        assert!(result.code.contains("name : string;"));
        // Last field should NOT have semicolon
        let lines: Vec<&str> = result.code.lines().collect();
        let age_line = lines.iter().find(|l| l.contains("age : Z")).unwrap();
        assert!(!age_line.ends_with(';'));
    }

    #[test]
    fn test_compile_module_with_theorem() {
        let compiler = CoqCompiler::new("Test");
        let spec = make_typed_spec(
            vec![],
            vec![Property::Theorem(Theorem {
                name: "thm".to_string(),
                body: Expr::Bool(true),
            })],
        );
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("(* Theorem: thm *)"));
        assert!(result.code.contains("Theorem thm : true."));
    }

    #[test]
    fn test_compile_module_with_invariant() {
        let compiler = CoqCompiler::new("Test");
        let spec = make_typed_spec(
            vec![],
            vec![Property::Invariant(Invariant {
                name: "inv".to_string(),
                body: Expr::Bool(true),
            })],
        );
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("(* Invariant: inv *)"));
        assert!(result.code.contains("Lemma inv : true."));
    }

    #[test]
    fn test_compile_module_imports() {
        let compiler = CoqCompiler::new("Test");
        let spec = make_typed_spec(vec![], vec![]);
        let result = compiler.compile_module(&spec);
        assert!(result.imports.contains(&"Coq.ZArith.ZArith".to_string()));
        assert!(result.imports.contains(&"Coq.Lists.List".to_string()));
    }

    // ========== compile_to_coq tests ==========

    #[test]
    fn test_compile_to_coq() {
        let spec = make_typed_spec(vec![], vec![]);
        let result = compile_to_coq(&spec);
        assert_eq!(result.backend, "Coq");
        assert_eq!(result.module_name, Some("USLSpec".to_string()));
    }
}

// =========================================================================
// Kani proofs for Coq compiler correctness
// =========================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::ast::{BinaryOp, ComparisonOp, Expr, Type};

    /// Prove that compile_expr never produces empty output for integer literals.
    #[kani::proof]
    fn verify_coq_compile_expr_int_nonempty() {
        let compiler = CoqCompiler::new("Test");
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[kani::proof]
    fn verify_coq_compile_expr_bool_nonempty() {
        let compiler = CoqCompiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Coq uses lowercase true/false
        assert!(result == "true" || result == "false");
    }

    /// Prove that compile_type always produces non-empty output for named types.
    #[kani::proof]
    fn verify_coq_compile_type_named_nonempty() {
        let compiler = CoqCompiler::new("Test");
        let ty = Type::Named("T".to_string());
        let result = compiler.compile_type(&ty);
        assert!(!result.is_empty());
    }

    /// Prove that compile_type maps Int to Z correctly.
    #[kani::proof]
    fn verify_coq_compile_type_int() {
        let compiler = CoqCompiler::new("Test");
        let ty = Type::Named("Int".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "Z");
    }

    /// Prove that compile_type maps Bool to bool correctly.
    #[kani::proof]
    fn verify_coq_compile_type_bool() {
        let compiler = CoqCompiler::new("Test");
        let ty = Type::Named("Bool".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "bool");
    }

    /// Prove that comparison operators compile to valid Coq syntax.
    #[kani::proof]
    fn verify_coq_comparison_valid() {
        let compiler = CoqCompiler::new("Test");
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
    fn verify_coq_binary_ops_nonempty() {
        let compiler = CoqCompiler::new("Test");
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
    fn verify_coq_implies_nonempty() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::Implies(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that and compiles to non-empty output.
    #[kani::proof]
    fn verify_coq_and_nonempty() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that or compiles to non-empty output.
    #[kani::proof]
    fn verify_coq_or_nonempty() {
        let compiler = CoqCompiler::new("Test");
        let expr = Expr::Or(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }
}
