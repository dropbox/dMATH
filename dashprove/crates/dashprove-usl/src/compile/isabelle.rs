//! Isabelle/HOL backend compiler
//!
//! Compiles USL specifications to Isabelle/HOL theories.

use crate::ast::{BinaryOp, ComparisonOp, Expr, Invariant, Property, Theorem, Type};
use crate::typecheck::TypedSpec;

use super::CompiledSpec;

/// Isabelle/HOL compiler
pub struct IsabelleCompiler {
    theory_name: String,
}

impl IsabelleCompiler {
    /// Create a new Isabelle compiler with the given theory name
    #[must_use]
    pub fn new(theory_name: &str) -> Self {
        Self {
            theory_name: theory_name.to_string(),
        }
    }

    /// Compile an expression to Isabelle/HOL syntax
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn compile_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Var(name) => name.clone(),
            Expr::Int(n) => n.to_string(),
            Expr::Float(f) => f.to_string(),
            Expr::String(s) => format!("''{s}''"),
            Expr::Bool(b) => if *b { "True" } else { "False" }.to_string(),

            Expr::ForAll { var, ty, body } => {
                let ty_str = ty
                    .as_ref()
                    .map(|t| format!("::'{}", self.compile_type(t)))
                    .unwrap_or_default();
                format!(
                    "(\\<forall>{}{} . {})",
                    var,
                    ty_str,
                    self.compile_expr(body)
                )
            }
            Expr::Exists { var, ty, body } => {
                let ty_str = ty
                    .as_ref()
                    .map(|t| format!("::'{}", self.compile_type(t)))
                    .unwrap_or_default();
                format!(
                    "(\\<exists>{}{} . {})",
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
                format!(
                    "(\\<forall>{} \\<in> {} . {})",
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
                    "(\\<exists>{} \\<in> {} . {})",
                    var,
                    self.compile_expr(collection),
                    self.compile_expr(body)
                )
            }

            Expr::Implies(lhs, rhs) => {
                format!(
                    "({} \\<longrightarrow> {})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::And(lhs, rhs) => {
                format!(
                    "({} \\<and> {})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Or(lhs, rhs) => {
                format!(
                    "({} \\<or> {})",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Not(e) => format!("(\\<not> {})", self.compile_expr(e)),

            Expr::Compare(lhs, op, rhs) => {
                let op_str = match op {
                    ComparisonOp::Eq => "=",
                    ComparisonOp::Ne => "\\<noteq>",
                    ComparisonOp::Lt => "<",
                    ComparisonOp::Le => "\\<le>",
                    ComparisonOp::Gt => ">",
                    ComparisonOp::Ge => "\\<ge>",
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
                    BinaryOp::Div => "div",
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

    /// Compile a type to Isabelle/HOL syntax
    #[must_use]
    pub fn compile_type(&self, ty: &Type) -> String {
        match ty {
            Type::Named(name) => match name.as_str() {
                "Int" => "int".to_string(),
                "Bool" => "bool".to_string(),
                "Nat" => "nat".to_string(),
                "String" => "string".to_string(),
                _ => name.to_lowercase(),
            },
            Type::Set(inner) => format!("{} set", self.compile_type(inner)),
            Type::List(inner) => format!("{} list", self.compile_type(inner)),
            Type::Map(k, v) => {
                format!(
                    "{} \\<Rightarrow> {}",
                    self.compile_type(k),
                    self.compile_type(v)
                )
            }
            Type::Relation(a, b) => {
                format!(
                    "({} \\<times> {}) set",
                    self.compile_type(a),
                    self.compile_type(b)
                )
            }
            Type::Function(a, b) => {
                format!(
                    "{} \\<Rightarrow> {}",
                    self.compile_type(a),
                    self.compile_type(b)
                )
            }
            Type::Result(inner) => format!("{} option", self.compile_type(inner)),
            Type::Unit => "unit".to_string(),
            Type::Graph(n, e) => {
                // Isabelle graph as record with nodes set and edges relation
                format!(
                    "\\<lparr> nodes :: {} set, edges :: ({} \\<times> {} \\<times> {}) set \\<rparr>",
                    self.compile_type(n),
                    self.compile_type(n),
                    self.compile_type(e),
                    self.compile_type(n)
                )
            }
            Type::Path(n) => format!("{} list", self.compile_type(n)),
        }
    }

    /// Compile a theorem to Isabelle lemma
    #[must_use]
    pub fn compile_theorem(&self, thm: &Theorem) -> String {
        format!(
            "lemma {} : \"{}\"\n  sorry",
            thm.name,
            self.compile_expr(&thm.body)
        )
    }

    /// Compile an invariant to Isabelle lemma
    #[must_use]
    pub fn compile_invariant(&self, inv: &Invariant) -> String {
        format!(
            "lemma {} : \"{}\"\n  sorry",
            inv.name,
            self.compile_expr(&inv.body)
        )
    }

    /// Generate complete Isabelle theory from spec
    #[must_use]
    pub fn compile_module(&self, typed_spec: &TypedSpec) -> CompiledSpec {
        let mut sections = Vec::new();

        // Theory header
        sections.push(format!("theory {}", self.theory_name));
        sections.push("  imports Main".to_string());
        sections.push("begin".to_string());
        sections.push(String::new());

        // Compile type definitions as datatypes
        for type_def in &typed_spec.spec.types {
            sections.push(format!("record {} =", type_def.name.to_lowercase()));
            for field in &type_def.fields {
                sections.push(format!(
                    "  {} :: \"{}\"",
                    field.name,
                    self.compile_type(&field.ty)
                ));
            }
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

        // Theory footer
        sections.push("end".to_string());

        CompiledSpec {
            backend: "Isabelle".to_string(),
            code: sections.join("\n"),
            module_name: Some(self.theory_name.clone()),
            imports: vec!["Main".to_string()],
        }
    }
}

/// Compile to Isabelle/HOL
#[must_use]
pub fn compile_to_isabelle(spec: &TypedSpec) -> CompiledSpec {
    let compiler = IsabelleCompiler::new("USLSpec");
    compiler.compile_module(spec)
}

// ========== Kani Proofs ==========

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::ast::{BinaryOp, ComparisonOp, Expr, Type};

    /// Prove that compile_expr never produces empty output for integer literals.
    #[kani::proof]
    fn verify_isabelle_compile_expr_int_nonempty() {
        let compiler = IsabelleCompiler::new("Test");
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[kani::proof]
    fn verify_isabelle_compile_expr_bool_nonempty() {
        let compiler = IsabelleCompiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Isabelle uses capitalized True/False
        assert!(result == "True" || result == "False");
    }

    /// Prove that compile_type always produces non-empty output for named types.
    #[kani::proof]
    fn verify_isabelle_compile_type_nonempty() {
        let compiler = IsabelleCompiler::new("Test");
        let ty = Type::Named("T".to_string());
        let result = compiler.compile_type(&ty);
        assert!(!result.is_empty());
    }

    /// Prove that comparison operators compile to valid Isabelle syntax.
    #[kani::proof]
    fn verify_isabelle_comparison_valid() {
        let compiler = IsabelleCompiler::new("Test");
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
        // All comparison results should start with "(" in Isabelle
        assert!(result.starts_with('('));
    }

    /// Prove that binary operators compile to valid Isabelle syntax.
    #[kani::proof]
    fn verify_isabelle_binary_ops_nonempty() {
        let compiler = IsabelleCompiler::new("Test");
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
    fn verify_isabelle_compile_type_unit() {
        let compiler = IsabelleCompiler::new("Test");
        let ty = Type::Unit;
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "unit");
    }

    /// Prove that Int type maps to int in Isabelle.
    #[kani::proof]
    fn verify_isabelle_compile_type_int() {
        let compiler = IsabelleCompiler::new("Test");
        let ty = Type::Named("Int".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "int");
    }

    /// Prove that Bool type maps to bool in Isabelle.
    #[kani::proof]
    fn verify_isabelle_compile_type_bool() {
        let compiler = IsabelleCompiler::new("Test");
        let ty = Type::Named("Bool".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "bool");
    }

    /// Prove that logical And expressions compile correctly.
    #[kani::proof]
    fn verify_isabelle_and_nonempty() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::And(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        assert!(result.contains("\\<and>"));
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
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Var("x".to_string())), "x");
    }

    #[test]
    fn test_compile_expr_int() {
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Int(42)), "42");
        assert_eq!(compiler.compile_expr(&Expr::Int(-5)), "-5");
    }

    #[test]
    fn test_compile_expr_float() {
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Float(2.71)), "2.71");
    }

    #[test]
    fn test_compile_expr_string() {
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(
            compiler.compile_expr(&Expr::String("hello".to_string())),
            "''hello''"
        );
    }

    #[test]
    fn test_compile_expr_bool() {
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(compiler.compile_expr(&Expr::Bool(true)), "True");
        assert_eq!(compiler.compile_expr(&Expr::Bool(false)), "False");
    }

    #[test]
    fn test_compile_expr_forall_with_type() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::ForAll {
            var: "x".to_string(),
            ty: Some(Type::Named("Int".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert!(compiler.compile_expr(&expr).contains("\\<forall>x::'int"));
    }

    #[test]
    fn test_compile_expr_forall_no_type() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::ForAll {
            var: "x".to_string(),
            ty: None,
            body: Box::new(Expr::Bool(true)),
        };
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("\\<forall>x"));
        assert!(result.contains("True"));
    }

    #[test]
    fn test_compile_expr_exists_with_type() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::Exists {
            var: "x".to_string(),
            ty: Some(Type::Named("Nat".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        assert!(compiler.compile_expr(&expr).contains("\\<exists>x::'nat"));
    }

    #[test]
    fn test_compile_expr_forall_in() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::ForAllIn {
            var: "x".to_string(),
            collection: Box::new(Expr::Var("S".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("\\<forall>x \\<in> S"));
    }

    #[test]
    fn test_compile_expr_exists_in() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::ExistsIn {
            var: "x".to_string(),
            collection: Box::new(Expr::Var("S".to_string())),
            body: Box::new(Expr::Bool(true)),
        };
        let result = compiler.compile_expr(&expr);
        assert!(result.contains("\\<exists>x \\<in> S"));
    }

    #[test]
    fn test_compile_expr_implies() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::Implies(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert!(compiler.compile_expr(&expr).contains("\\<longrightarrow>"));
    }

    #[test]
    fn test_compile_expr_and() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::And(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert!(compiler.compile_expr(&expr).contains("\\<and>"));
    }

    #[test]
    fn test_compile_expr_or() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::Or(
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        assert!(compiler.compile_expr(&expr).contains("\\<or>"));
    }

    #[test]
    fn test_compile_expr_not() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::Not(Box::new(Expr::Var("a".to_string())));
        assert!(compiler.compile_expr(&expr).contains("\\<not>"));
    }

    #[test]
    fn test_compile_expr_compare_all_ops() {
        let compiler = IsabelleCompiler::new("Test");
        let ops = [
            (ComparisonOp::Eq, "="),
            (ComparisonOp::Ne, "\\<noteq>"),
            (ComparisonOp::Lt, "<"),
            (ComparisonOp::Le, "\\<le>"),
            (ComparisonOp::Gt, ">"),
            (ComparisonOp::Ge, "\\<ge>"),
        ];
        for (op, expected) in ops {
            let expr = Expr::Compare(
                Box::new(Expr::Var("x".to_string())),
                op,
                Box::new(Expr::Var("y".to_string())),
            );
            let result = compiler.compile_expr(&expr);
            assert!(
                result.contains(expected),
                "Expected {} in {}",
                expected,
                result
            );
        }
    }

    #[test]
    fn test_compile_expr_binary_all_ops() {
        let compiler = IsabelleCompiler::new("Test");
        let ops = [
            (BinaryOp::Add, "+"),
            (BinaryOp::Sub, "-"),
            (BinaryOp::Mul, "*"),
            (BinaryOp::Div, "div"),
            (BinaryOp::Mod, "mod"),
        ];
        for (op, expected) in ops {
            let expr = Expr::Binary(
                Box::new(Expr::Var("x".to_string())),
                op,
                Box::new(Expr::Var("y".to_string())),
            );
            let result = compiler.compile_expr(&expr);
            assert!(
                result.contains(expected),
                "Expected {} in {}",
                expected,
                result
            );
        }
    }

    #[test]
    fn test_compile_expr_neg() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::Neg(Box::new(Expr::Var("x".to_string())));
        assert_eq!(compiler.compile_expr(&expr), "(- x)");
    }

    #[test]
    fn test_compile_expr_app_no_args() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::App("f".to_string(), vec![]);
        assert_eq!(compiler.compile_expr(&expr), "f");
    }

    #[test]
    fn test_compile_expr_app_with_args() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::App(
            "f".to_string(),
            vec![Expr::Var("x".to_string()), Expr::Int(1)],
        );
        assert_eq!(compiler.compile_expr(&expr), "(f x 1)");
    }

    #[test]
    fn test_compile_expr_method_call_no_args() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".to_string())),
            method: "size".to_string(),
            args: vec![],
        };
        assert_eq!(compiler.compile_expr(&expr), "(size obj)");
    }

    #[test]
    fn test_compile_expr_method_call_with_args() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::MethodCall {
            receiver: Box::new(Expr::Var("obj".to_string())),
            method: "get".to_string(),
            args: vec![Expr::Int(1)],
        };
        assert_eq!(compiler.compile_expr(&expr), "(get obj 1)");
    }

    #[test]
    fn test_compile_expr_field_access() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::FieldAccess(Box::new(Expr::Var("obj".to_string())), "field".to_string());
        assert_eq!(compiler.compile_expr(&expr), "(field obj)");
    }

    // ========== compile_type tests ==========

    #[test]
    fn test_compile_type_int_maps_to_int() {
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("Int".to_string())),
            "int"
        );
    }

    #[test]
    fn test_compile_type_bool_maps_to_bool() {
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("Bool".to_string())),
            "bool"
        );
    }

    #[test]
    fn test_compile_type_nat_maps_to_nat() {
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("Nat".to_string())),
            "nat"
        );
    }

    #[test]
    fn test_compile_type_string_maps_to_string() {
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("String".to_string())),
            "string"
        );
    }

    #[test]
    fn test_compile_type_custom_name() {
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Named("Person".to_string())),
            "person"
        );
    }

    #[test]
    fn test_compile_type_set() {
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Set(Box::new(Type::Named("Int".to_string())))),
            "int set"
        );
    }

    #[test]
    fn test_compile_type_list() {
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::List(Box::new(Type::Named("Int".to_string())))),
            "int list"
        );
    }

    #[test]
    fn test_compile_type_map() {
        let compiler = IsabelleCompiler::new("Test");
        let result = compiler.compile_type(&Type::Map(
            Box::new(Type::Named("String".to_string())),
            Box::new(Type::Named("Int".to_string())),
        ));
        assert!(result.contains("\\<Rightarrow>"));
    }

    #[test]
    fn test_compile_type_relation() {
        let compiler = IsabelleCompiler::new("Test");
        let result = compiler.compile_type(&Type::Relation(
            Box::new(Type::Named("Int".to_string())),
            Box::new(Type::Named("Bool".to_string())),
        ));
        assert!(result.contains("\\<times>"));
        assert!(result.contains("set"));
    }

    #[test]
    fn test_compile_type_function() {
        let compiler = IsabelleCompiler::new("Test");
        let result = compiler.compile_type(&Type::Function(
            Box::new(Type::Named("Int".to_string())),
            Box::new(Type::Named("Bool".to_string())),
        ));
        assert!(result.contains("\\<Rightarrow>"));
    }

    #[test]
    fn test_compile_type_result() {
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(
            compiler.compile_type(&Type::Result(Box::new(Type::Named("Int".to_string())))),
            "int option"
        );
    }

    #[test]
    fn test_compile_type_unit() {
        let compiler = IsabelleCompiler::new("Test");
        assert_eq!(compiler.compile_type(&Type::Unit), "unit");
    }

    // ========== compile_theorem and compile_invariant tests ==========

    #[test]
    fn test_compile_theorem() {
        let compiler = IsabelleCompiler::new("Test");
        let thm = Theorem {
            name: "my_theorem".to_string(),
            body: Expr::Bool(true),
        };
        let result = compiler.compile_theorem(&thm);
        assert!(result.contains("lemma my_theorem"));
        assert!(result.contains("True"));
        assert!(result.contains("sorry"));
    }

    #[test]
    fn test_compile_invariant() {
        let compiler = IsabelleCompiler::new("Test");
        let inv = Invariant {
            name: "my_inv".to_string(),
            body: Expr::Bool(false),
        };
        let result = compiler.compile_invariant(&inv);
        assert!(result.contains("lemma my_inv"));
        assert!(result.contains("False"));
        assert!(result.contains("sorry"));
    }

    // ========== compile_module tests ==========

    #[test]
    fn test_compile_module_header() {
        let compiler = IsabelleCompiler::new("MyTheory");
        let spec = make_typed_spec(vec![], vec![]);
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("theory MyTheory"));
        assert!(result.code.contains("imports Main"));
        assert!(result.code.contains("begin"));
        assert!(result.code.contains("end"));
        assert_eq!(result.backend, "Isabelle");
        assert_eq!(result.module_name, Some("MyTheory".to_string()));
    }

    #[test]
    fn test_compile_module_with_type() {
        let compiler = IsabelleCompiler::new("Test");
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
        assert!(result.code.contains("record person ="));
        assert!(result.code.contains("name :: \"string\""));
        assert!(result.code.contains("age :: \"int\""));
    }

    #[test]
    fn test_compile_module_with_theorem() {
        let compiler = IsabelleCompiler::new("Test");
        let spec = make_typed_spec(
            vec![],
            vec![Property::Theorem(Theorem {
                name: "thm".to_string(),
                body: Expr::Bool(true),
            })],
        );
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("(* Theorem: thm *)"));
        assert!(result.code.contains("lemma thm"));
    }

    #[test]
    fn test_compile_module_with_invariant() {
        let compiler = IsabelleCompiler::new("Test");
        let spec = make_typed_spec(
            vec![],
            vec![Property::Invariant(Invariant {
                name: "inv".to_string(),
                body: Expr::Bool(true),
            })],
        );
        let result = compiler.compile_module(&spec);
        assert!(result.code.contains("(* Invariant: inv *)"));
        assert!(result.code.contains("lemma inv"));
    }

    #[test]
    fn test_compile_module_imports() {
        let compiler = IsabelleCompiler::new("Test");
        let spec = make_typed_spec(vec![], vec![]);
        let result = compiler.compile_module(&spec);
        assert!(result.imports.contains(&"Main".to_string()));
    }

    // ========== compile_to_isabelle tests ==========

    #[test]
    fn test_compile_to_isabelle() {
        let spec = make_typed_spec(vec![], vec![]);
        let result = compile_to_isabelle(&spec);
        assert_eq!(result.backend, "Isabelle");
        assert_eq!(result.module_name, Some("USLSpec".to_string()));
    }
}

// =========================================================================
// Kani proofs for Isabelle compiler correctness
// =========================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::ast::{BinaryOp, ComparisonOp, Expr, Type};

    /// Prove that compile_expr never produces empty output for integer literals.
    #[kani::proof]
    fn verify_isabelle_compile_expr_int_nonempty() {
        let compiler = IsabelleCompiler::new("Test");
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[kani::proof]
    fn verify_isabelle_compile_expr_bool_nonempty() {
        let compiler = IsabelleCompiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Isabelle uses True/False
        assert!(result == "True" || result == "False");
    }

    /// Prove that compile_type always produces non-empty output for named types.
    #[kani::proof]
    fn verify_isabelle_compile_type_named_nonempty() {
        let compiler = IsabelleCompiler::new("Test");
        let ty = Type::Named("T".to_string());
        let result = compiler.compile_type(&ty);
        assert!(!result.is_empty());
    }

    /// Prove that compile_type maps Int to int correctly.
    #[kani::proof]
    fn verify_isabelle_compile_type_int() {
        let compiler = IsabelleCompiler::new("Test");
        let ty = Type::Named("Int".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "int");
    }

    /// Prove that compile_type maps Bool to bool correctly.
    #[kani::proof]
    fn verify_isabelle_compile_type_bool() {
        let compiler = IsabelleCompiler::new("Test");
        let ty = Type::Named("Bool".to_string());
        let result = compiler.compile_type(&ty);
        assert_eq!(result, "bool");
    }

    /// Prove that comparison operators compile to valid Isabelle syntax.
    #[kani::proof]
    fn verify_isabelle_comparison_valid() {
        let compiler = IsabelleCompiler::new("Test");
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
    fn verify_isabelle_binary_ops_nonempty() {
        let compiler = IsabelleCompiler::new("Test");
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
    fn verify_isabelle_implies_nonempty() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::Implies(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that and compiles to non-empty output.
    #[kani::proof]
    fn verify_isabelle_and_nonempty() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that or compiles to non-empty output.
    #[kani::proof]
    fn verify_isabelle_or_nonempty() {
        let compiler = IsabelleCompiler::new("Test");
        let expr = Expr::Or(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }
}
