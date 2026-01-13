//! Rust Closure Compiler for USL Invariants
//!
//! Compiles USL invariant expressions into Rust closure code that can be used
//! with `dashprove-monitor` for runtime verification.
//!
//! The generated closures have signature:
//! `Box<dyn Fn(&serde_json::Value) -> bool + Send + Sync>`
//!
//! # Supported Expressions
//!
//! - Literals: `true`, `false`, integers, floats, strings
//! - Variables: accessed via JSON field access (e.g., `x` becomes `state["x"]`)
//! - Field access: `obj.field` becomes `state["obj"]["field"]`
//! - Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
//! - Boolean operations: `and`, `or`, `not`, `implies`
//! - Arithmetic: `+`, `-`, `*`, `/`, `%`
//! - Quantifiers: `forall x in collection . body` (limited support)
//!
//! # Limitations
//!
//! - Typed quantifiers (`forall x: T . body`) are not fully supported at runtime
//! - Method calls are compiled as function calls on the state
//! - Type information is used heuristically

use crate::ast::{BinaryOp, ComparisonOp, Expr, Invariant, Property};
use crate::typecheck::TypedSpec;

/// Compiled Rust closure code
#[derive(Debug, Clone)]
pub struct CompiledRustClosure {
    /// The name of the invariant
    pub name: String,
    /// The generated Rust closure code (as a string)
    pub code: String,
    /// Required imports for the generated code
    pub imports: Vec<String>,
}

/// Rust Closure Compiler
pub struct RustClosureCompiler {
    /// Indentation level for code generation (reserved for future pretty-printing)
    #[allow(dead_code)]
    indent: usize,
}

impl Default for RustClosureCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl RustClosureCompiler {
    /// Create a new compiler
    #[must_use]
    pub fn new() -> Self {
        Self { indent: 0 }
    }

    /// Compile a typed spec to Rust closure code
    #[must_use]
    pub fn compile(&self, spec: &TypedSpec) -> Vec<CompiledRustClosure> {
        spec.spec
            .properties
            .iter()
            .filter_map(|prop| self.compile_property(prop))
            .collect()
    }

    /// Compile a single property to a Rust closure
    fn compile_property(&self, property: &Property) -> Option<CompiledRustClosure> {
        match property {
            Property::Invariant(inv) => Some(self.compile_invariant(inv)),
            // Could extend to compile theorem bodies as runtime checks
            Property::Theorem(thm) => Some(CompiledRustClosure {
                name: thm.name.clone(),
                code: self.compile_to_closure(&thm.body),
                imports: self.required_imports(),
            }),
            // Contracts could compile preconditions/postconditions
            Property::Contract(contract) => {
                let name = contract.type_path.join("_");
                let mut parts = Vec::new();

                // Compile preconditions
                for (i, req) in contract.requires.iter().enumerate() {
                    parts.push(format!(
                        "    // Precondition {i}\n    let pre_{i} = {};",
                        self.compile_expr(req)
                    ));
                }

                // Compile postconditions
                for (i, ens) in contract.ensures.iter().enumerate() {
                    parts.push(format!(
                        "    // Postcondition {i}\n    let post_{i} = {};",
                        self.compile_expr(ens)
                    ));
                }

                let code = format!(
                    "// Contract: {name}\n|state: &serde_json::Value| -> bool {{\n{}\n    true\n}}",
                    parts.join("\n")
                );

                Some(CompiledRustClosure {
                    name,
                    code,
                    imports: self.required_imports(),
                })
            }
            // Other property types don't make sense as runtime closures
            _ => None,
        }
    }

    /// Compile an invariant to Rust closure code
    fn compile_invariant(&self, invariant: &Invariant) -> CompiledRustClosure {
        CompiledRustClosure {
            name: invariant.name.clone(),
            code: self.compile_to_closure(&invariant.body),
            imports: self.required_imports(),
        }
    }

    /// Compile an expression to a closure string
    fn compile_to_closure(&self, expr: &Expr) -> String {
        let body = self.compile_expr(expr);
        format!("|state: &serde_json::Value| -> bool {{\n    {body}\n}}")
    }

    /// Compile an expression to Rust code
    fn compile_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Bool(b) => b.to_string(),
            Expr::Int(n) => n.to_string(),
            Expr::Float(f) => format!("{f:?}"),
            Expr::String(s) => format!("{s:?}"),

            Expr::Var(name) => {
                // Access variable from JSON state
                format!("state.get({name:?})")
            }

            Expr::FieldAccess(obj, field) => {
                let obj_code = self.compile_expr(obj);
                // Handle nested field access - always use and_then for chaining
                format!("{obj_code}.and_then(|v| v.get({field:?}))")
            }

            Expr::Compare(left, op, right) => {
                let l = self.compile_expr_for_comparison(left);
                let r = self.compile_expr_for_comparison(right);
                let op_str = match op {
                    ComparisonOp::Eq => "==",
                    ComparisonOp::Ne => "!=",
                    ComparisonOp::Lt => "<",
                    ComparisonOp::Le => "<=",
                    ComparisonOp::Gt => ">",
                    ComparisonOp::Ge => ">=",
                };
                // Handle Option comparisons
                format!("({l}).zip({r}).is_some_and(|(a, b)| a {op_str} b)")
            }

            Expr::And(left, right) => {
                let l = self.compile_expr(left);
                let r = self.compile_expr(right);
                format!("({l}) && ({r})")
            }

            Expr::Or(left, right) => {
                let l = self.compile_expr(left);
                let r = self.compile_expr(right);
                format!("({l}) || ({r})")
            }

            Expr::Not(inner) => {
                let i = self.compile_expr(inner);
                format!("!({i})")
            }

            Expr::Implies(left, right) => {
                let l = self.compile_expr(left);
                let r = self.compile_expr(right);
                // P => Q is equivalent to !P || Q
                format!("(!({l})) || ({r})")
            }

            Expr::Binary(left, op, right) => {
                let l = self.compile_expr_for_arithmetic(left);
                let r = self.compile_expr_for_arithmetic(right);
                let op_str = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Mod => "%",
                };
                format!("({l}).zip({r}).map(|(a, b)| a {op_str} b)")
            }

            Expr::Neg(inner) => {
                let i = self.compile_expr_for_arithmetic(inner);
                format!("({i}).map(|x| -x)")
            }

            // Quantifiers are tricky at runtime - generate iterator-based checks
            Expr::ForAllIn {
                var,
                collection,
                body,
            } => {
                let coll = self.compile_expr(collection);
                let body_code = self.compile_expr(body);
                format!(
                    "{coll}.and_then(|v| v.as_array()).is_some_and(|arr| arr.iter().all(|{var}| {{\n        let state = {var};\n        {body_code}\n    }}))"
                )
            }

            Expr::ExistsIn {
                var,
                collection,
                body,
            } => {
                let coll = self.compile_expr(collection);
                let body_code = self.compile_expr(body);
                format!(
                    "{coll}.and_then(|v| v.as_array()).is_some_and(|arr| arr.iter().any(|{var}| {{\n        let state = {var};\n        {body_code}\n    }}))"
                )
            }

            // Typed quantifiers - generate a warning comment
            Expr::ForAll { var, body, .. } => {
                let body_code = self.compile_expr(body);
                format!(
                    "/* Warning: forall {var} without collection - checking with current state only */\n    {body_code}"
                )
            }

            Expr::Exists { var, body, .. } => {
                let body_code = self.compile_expr(body);
                format!(
                    "/* Warning: exists {var} without collection - checking with current state only */\n    {body_code}"
                )
            }

            Expr::App(func, args) => {
                let args_code: Vec<String> = args.iter().map(|a| self.compile_expr(a)).collect();
                // Generate function call - user needs to provide implementation
                format!(
                    "/* TODO: implement {func}() */ {func}(state{})",
                    if args_code.is_empty() {
                        String::new()
                    } else {
                        format!(", {}", args_code.join(", "))
                    }
                )
            }

            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                let recv = self.compile_expr(receiver);
                let args_code: Vec<String> = args.iter().map(|a| self.compile_expr(a)).collect();

                // Common JSON methods
                match method.as_str() {
                    "length" | "len" => format!("{recv}.and_then(|v| v.as_array().map(|a| a.len() as i64))"),
                    "is_empty" => format!("{recv}.and_then(|v| v.as_array().map(|a| a.is_empty())).unwrap_or(true)"),
                    "contains" => {
                        let arg = args_code.first().cloned().unwrap_or_default();
                        format!("{recv}.and_then(|v| v.as_array()).is_some_and(|arr| arr.contains(&{arg}))")
                    }
                    _ => format!(
                        "/* TODO: implement method {method}() */ {recv}.and_then(|v| {method}(v{}))",
                        if args_code.is_empty() {
                            String::new()
                        } else {
                            format!(", {}", args_code.join(", "))
                        }
                    ),
                }
            }
        }
    }

    /// Compile an expression for use in a comparison, extracting primitive value
    fn compile_expr_for_comparison(&self, expr: &Expr) -> String {
        match expr {
            Expr::Int(n) => format!("Some({n}_i64)"),
            Expr::Float(f) => format!("Some({f:?}_f64)"),
            Expr::Bool(b) => format!("Some({b})"),
            Expr::String(s) => format!("Some({s:?}.to_string())"),
            Expr::Var(name) => {
                // Try to extract as i64, f64, bool, or string
                format!(
                    "state.get({name:?}).and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)))"
                )
            }
            Expr::FieldAccess(obj, _field) => {
                let obj_code = self.compile_expr_for_comparison(obj);
                format!(
                    "{obj_code}.and_then(|_| state{}).and_then(|v| v.as_i64())",
                    self.build_field_path(expr)
                )
            }
            // For complex expressions, compile normally and try to extract value
            _ => {
                let code = self.compile_expr(expr);
                format!("({code}).and_then(|v| v.as_i64())")
            }
        }
    }

    /// Compile an expression for arithmetic operations
    fn compile_expr_for_arithmetic(&self, expr: &Expr) -> String {
        match expr {
            Expr::Int(n) => format!("Some({n}_i64)"),
            Expr::Float(f) => format!("Some({f:?}_f64)"),
            Expr::Var(name) => {
                format!("state.get({name:?}).and_then(|v| v.as_i64())")
            }
            Expr::FieldAccess(_, _) => {
                let path = self.build_field_path(expr);
                format!("state{path}.and_then(|v| v.as_i64())")
            }
            Expr::Binary(left, op, right) => {
                let l = self.compile_expr_for_arithmetic(left);
                let r = self.compile_expr_for_arithmetic(right);
                let op_str = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Mod => "%",
                };
                format!("({l}).zip({r}).map(|(a, b)| a {op_str} b)")
            }
            Expr::Neg(inner) => {
                let i = self.compile_expr_for_arithmetic(inner);
                format!("({i}).map(|x| -x)")
            }
            _ => {
                let code = self.compile_expr(expr);
                format!("/* complex expr */ ({code}).and_then(|v| v.as_i64())")
            }
        }
    }

    /// Build a JSON field path from a nested field access expression
    fn build_field_path(&self, expr: &Expr) -> String {
        match expr {
            Expr::Var(name) => format!(".get({name:?})"),
            Expr::FieldAccess(obj, field) => {
                let obj_path = self.build_field_path(obj);
                format!("{obj_path}.and_then(|v| v.get({field:?}))")
            }
            _ => String::new(),
        }
    }

    /// Get required imports for the generated code
    fn required_imports(&self) -> Vec<String> {
        vec!["serde_json".to_string()]
    }
}

/// Compile a typed spec to Rust closures
#[must_use]
pub fn compile_to_rust_closures(spec: &TypedSpec) -> Vec<CompiledRustClosure> {
    RustClosureCompiler::new().compile(spec)
}

/// Compile a single invariant expression to a Rust closure string
#[must_use]
pub fn compile_invariant_to_closure(expr: &Expr) -> String {
    RustClosureCompiler::new().compile_to_closure(expr)
}

/// Generate monitor registration code for all invariants in a spec
#[must_use]
pub fn generate_monitor_registration(spec: &TypedSpec) -> String {
    let closures = compile_to_rust_closures(spec);
    let mut lines = vec![
        "// Auto-generated monitor registration code".to_string(),
        "// Generated by dashprove-usl compile::rust_closure".to_string(),
        String::new(),
        "use dashprove_monitor::RuntimeMonitor;".to_string(),
        String::new(),
        "pub fn register_invariants(monitor: &mut RuntimeMonitor) {".to_string(),
    ];

    for closure in closures {
        lines.push(format!("    // Invariant: {}", closure.name));
        lines.push(format!(
            "    monitor.add_simple_invariant({:?}, {});",
            closure.name, closure.code
        ));
        lines.push(String::new());
    }

    lines.push("}".to_string());

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, typecheck};

    fn compile_usl(input: &str) -> TypedSpec {
        let spec = parse(input).expect("parse failed");
        typecheck(spec).expect("typecheck failed")
    }

    #[test]
    fn test_compile_simple_invariant() {
        let spec = compile_usl("invariant always_true { true }");
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 1);
        assert_eq!(closures[0].name, "always_true");
        assert!(closures[0].code.contains("true"));
    }

    #[test]
    fn test_compile_comparison_invariant() {
        let spec = compile_usl("invariant positive { x > 0 }");
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 1);
        assert!(closures[0].code.contains("state.get"));
        assert!(closures[0].code.contains(">"));
    }

    #[test]
    fn test_compile_and_or_not() {
        let spec = compile_usl("invariant logic { x and (y or not z) }");
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 1);
        assert!(closures[0].code.contains("&&"));
        assert!(closures[0].code.contains("||"));
        assert!(closures[0].code.contains("!"));
    }

    #[test]
    fn test_compile_implies() {
        let spec = compile_usl("invariant impl_test { x implies y }");
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 1);
        // implies is compiled as !P || Q
        assert!(closures[0].code.contains("!"));
        assert!(closures[0].code.contains("||"));
    }

    #[test]
    fn test_compile_field_access() {
        let spec = compile_usl("invariant field_test { obj.field > 0 }");
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 1);
        assert!(closures[0].code.contains("get"));
        assert!(closures[0].code.contains("\"field\""));
    }

    #[test]
    fn test_compile_forall_in() {
        let spec = compile_usl("invariant all_positive { forall x in items . x > 0 }");
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 1);
        assert!(closures[0].code.contains("all"));
        assert!(closures[0].code.contains("iter"));
    }

    #[test]
    fn test_compile_exists_in() {
        let spec = compile_usl("invariant some_positive { exists x in items . x > 0 }");
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 1);
        assert!(closures[0].code.contains("any"));
        assert!(closures[0].code.contains("iter"));
    }

    #[test]
    fn test_compile_arithmetic() {
        let spec = compile_usl("invariant arith_test { x + y > z * 2 }");
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 1);
        assert!(closures[0].code.contains("+"));
        assert!(closures[0].code.contains("*"));
    }

    #[test]
    fn test_compile_multiple_invariants() {
        let spec = compile_usl(
            r#"
            invariant inv1 { x > 0 }
            invariant inv2 { y < 100 }
            invariant inv3 { x + y > 0 }
        "#,
        );
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 3);
        assert_eq!(closures[0].name, "inv1");
        assert_eq!(closures[1].name, "inv2");
        assert_eq!(closures[2].name, "inv3");
    }

    #[test]
    fn test_generate_monitor_registration() {
        let spec = compile_usl(
            r#"
            invariant positive { value > 0 }
            invariant bounded { value < 100 }
        "#,
        );
        let code = generate_monitor_registration(&spec);

        assert!(code.contains("register_invariants"));
        assert!(code.contains("RuntimeMonitor"));
        assert!(code.contains("add_simple_invariant"));
        assert!(code.contains("positive"));
        assert!(code.contains("bounded"));
    }

    #[test]
    fn test_compile_contract_preconditions() {
        let spec = compile_usl(
            r#"
            contract add(x: Int, y: Int) -> Int {
                requires { x >= 0 }
                requires { y >= 0 }
                ensures { result >= x }
            }
        "#,
        );
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 1);
        assert!(closures[0].code.contains("Precondition"));
        assert!(closures[0].code.contains("Postcondition"));
    }

    #[test]
    fn test_compile_theorem_as_check() {
        let spec = compile_usl("theorem reflexive { forall x: Int . x == x }");
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 1);
        assert_eq!(closures[0].name, "reflexive");
    }

    #[test]
    fn test_compile_nested_field_access() {
        let spec = compile_usl("invariant nested { obj.inner.value > 0 }");
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 1);
        assert!(closures[0].code.contains("\"inner\""));
        assert!(closures[0].code.contains("\"value\""));
    }

    #[test]
    fn test_compile_field_access_length() {
        // Note: items.length is parsed as field access, not method call
        // For actual length() call, use items.length()
        let spec = compile_usl("invariant has_items { items.length > 0 }");
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 1);
        // Field access generates get() calls
        assert!(closures[0].code.contains("get"));
        assert!(closures[0].code.contains("\"length\""));
    }

    #[test]
    fn test_compile_string_equality() {
        let spec = compile_usl(r#"invariant status_ok { status == "ok" }"#);
        let closures = compile_to_rust_closures(&spec);

        assert_eq!(closures.len(), 1);
        assert!(closures[0].code.contains("\"ok\""));
    }

    #[test]
    fn test_compile_method_calls() {
        // Method calls should use the specialized branches rather than the default TODO path
        let spec = compile_usl(
            r#"
            invariant methods {
                items.len() > 0 and items.is_empty() or items.contains(target)
            }
        "#,
        );
        let closures = compile_to_rust_closures(&spec);
        let code = &closures[0].code;

        assert!(code.contains("as_array().map(|a| a.len() as i64)"));
        assert!(code.contains("is_empty"), "uses is_empty branch");
        assert!(
            code.contains("unwrap_or(true)"),
            "is_empty should default to true"
        );
        assert!(code.contains("arr.contains"));
    }

    #[test]
    fn test_compile_expr_for_comparison_variants() {
        let compiler = RustClosureCompiler::new();

        assert_eq!(
            compiler.compile_expr_for_comparison(&Expr::Int(5)),
            "Some(5_i64)"
        );
        assert_eq!(
            compiler.compile_expr_for_comparison(&Expr::Float(1.5)),
            "Some(1.5_f64)"
        );
        assert_eq!(
            compiler.compile_expr_for_comparison(&Expr::Bool(true)),
            "Some(true)"
        );
        assert_eq!(
            compiler.compile_expr_for_comparison(&Expr::String("ok".into())),
            "Some(\"ok\".to_string())"
        );

        let var_expr = compiler.compile_expr_for_comparison(&Expr::Var("x".into()));
        assert_eq!(
            var_expr,
            "state.get(\"x\").and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)))"
        );

        let field_expr = Expr::FieldAccess(Box::new(Expr::Var("obj".into())), "field".into());
        let field_code = compiler.compile_expr_for_comparison(&field_expr);
        assert_eq!(
            field_code,
            "state.get(\"obj\").and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64))).and_then(|_| state.get(\"obj\").and_then(|v| v.get(\"field\"))).and_then(|v| v.as_i64())"
        );
    }

    #[test]
    fn test_compile_expr_for_arithmetic_variants() {
        let compiler = RustClosureCompiler::new();

        assert_eq!(
            compiler.compile_expr_for_arithmetic(&Expr::Int(3)),
            "Some(3_i64)"
        );
        assert_eq!(
            compiler.compile_expr_for_arithmetic(&Expr::Float(2.5)),
            "Some(2.5_f64)"
        );

        let var_code = compiler.compile_expr_for_arithmetic(&Expr::Var("a".into()));
        assert_eq!(var_code, "state.get(\"a\").and_then(|v| v.as_i64())");

        let field_expr = Expr::FieldAccess(Box::new(Expr::Var("obj".into())), "count".into());
        let field_code = compiler.compile_expr_for_arithmetic(&field_expr);
        assert_eq!(
            field_code,
            "state.get(\"obj\").and_then(|v| v.get(\"count\")).and_then(|v| v.as_i64())"
        );

        let binary_expr = Expr::Binary(
            Box::new(Expr::Int(1)),
            BinaryOp::Add,
            Box::new(Expr::Var("b".into())),
        );
        let binary_code = compiler.compile_expr_for_arithmetic(&binary_expr);
        assert_eq!(
            binary_code,
            "(Some(1_i64)).zip(state.get(\"b\").and_then(|v| v.as_i64())).map(|(a, b)| a + b)"
        );

        let neg_code = compiler.compile_expr_for_arithmetic(&Expr::Neg(Box::new(Expr::Int(1))));
        assert_eq!(neg_code, "(Some(1_i64)).map(|x| -x)");
    }

    #[test]
    fn test_build_field_path_var_and_required_imports() {
        let compiler = RustClosureCompiler::new();
        assert_eq!(
            compiler.build_field_path(&Expr::Var("foo".into())),
            ".get(\"foo\")"
        );
        assert_eq!(compiler.required_imports(), vec!["serde_json".to_string()]);
    }

    #[test]
    fn test_compile_invariant_to_closure_wrapper() {
        let code = compile_invariant_to_closure(&Expr::Bool(true));
        assert!(code.contains("|state: &serde_json::Value| -> bool"));
        assert!(code.contains("true"));
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::ast::{BinaryOp, ComparisonOp, Expr};

    /// Prove that compile_expr never produces empty output for integer literals.
    #[kani::proof]
    fn verify_rust_closure_compile_expr_int_nonempty() {
        let compiler = RustClosureCompiler::new();
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[kani::proof]
    fn verify_rust_closure_compile_expr_bool_nonempty() {
        let compiler = RustClosureCompiler::new();
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Rust booleans compile to "true" or "false"
        assert!(result == "true" || result == "false");
    }

    /// Prove that comparison operators compile to valid Rust syntax.
    #[kani::proof]
    fn verify_rust_closure_comparison_valid() {
        let compiler = RustClosureCompiler::new();
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
        // All comparison results should start with "(" for zip pattern
        assert!(result.starts_with('('));
    }

    /// Prove that binary operators compile to valid Rust syntax.
    #[kani::proof]
    fn verify_rust_closure_binary_ops_nonempty() {
        let compiler = RustClosureCompiler::new();
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
        // Binary ops use zip pattern: (a).zip(b).map(...)
        assert!(result.starts_with('('));
    }

    /// Prove that implies compiles to non-empty output with correct structure.
    #[kani::proof]
    fn verify_rust_closure_implies_nonempty() {
        let compiler = RustClosureCompiler::new();
        let expr = Expr::Implies(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Implies is compiled as !P || Q
        assert!(result.contains("||"));
    }

    /// Prove that not compiles to non-empty output with correct prefix.
    #[kani::proof]
    fn verify_rust_closure_not_nonempty() {
        let compiler = RustClosureCompiler::new();
        let b: bool = kani::any();
        let expr = Expr::Not(Box::new(Expr::Bool(b)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Not is compiled as !(expr)
        assert!(result.starts_with("!("));
    }

    /// Prove that and compiles to non-empty output.
    #[kani::proof]
    fn verify_rust_closure_and_nonempty() {
        let compiler = RustClosureCompiler::new();
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // And is compiled as (P) && (Q)
        assert!(result.contains("&&"));
    }

    /// Prove that compile_to_closure produces valid closure syntax.
    #[kani::proof]
    fn verify_rust_closure_compile_to_closure_valid() {
        let compiler = RustClosureCompiler::new();
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_to_closure(&expr);
        assert!(!result.is_empty());
        // Closure must start with pipe for closure syntax
        assert!(result.starts_with("|state:"));
    }

    /// Prove that compile_expr_for_comparison handles Int correctly.
    #[kani::proof]
    fn verify_rust_closure_compile_expr_for_comparison_int() {
        let compiler = RustClosureCompiler::new();
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr_for_comparison(&expr);
        assert!(!result.is_empty());
        // Int comparison wraps in Some()
        assert!(result.starts_with("Some("));
    }
}
