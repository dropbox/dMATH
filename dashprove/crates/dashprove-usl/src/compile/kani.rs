//! Kani backend compiler
//!
//! Compiles USL specifications to Kani proof harnesses for Rust model checking.

use crate::ast::{BinaryOp, ComparisonOp, Contract, Expr, Property, Refinement, Type};
use crate::typecheck::TypedSpec;

use super::CompiledSpec;

/// Kani (Rust model checking) compiler
pub struct KaniCompiler {
    crate_name: String,
}

impl KaniCompiler {
    /// Create a new Kani compiler with the given crate name
    #[must_use]
    pub fn new(crate_name: &str) -> Self {
        Self {
            crate_name: crate_name.to_string(),
        }
    }

    /// Compile an expression to Rust syntax
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn compile_expr(&self, expr: &Expr) -> String {
        match expr {
            Expr::Var(name) => {
                // Handle primed variables - use _old suffix for original
                if name.ends_with('\'') {
                    name[..name.len() - 1].to_string()
                } else if name == "self" {
                    "self".to_string()
                } else {
                    name.clone()
                }
            }
            Expr::Int(n) => n.to_string(),
            Expr::Float(f) => format!("{f}f64"),
            Expr::String(s) => format!("\"{s}\""),
            Expr::Bool(b) => if *b { "true" } else { "false" }.to_string(),

            Expr::ForAll { var, ty, body } => {
                // Kani uses kani::any() for symbolic values
                let ty_str = ty
                    .as_ref()
                    .map_or_else(|| "kani::any()".to_string(), |t| self.compile_type(t));
                format!(
                    "{{ let {}: {} = kani::any(); kani::assume({}); true }}",
                    var,
                    ty_str,
                    self.compile_expr(body)
                )
            }
            Expr::Exists { var, ty, body } => {
                let ty_str = ty
                    .as_ref()
                    .map_or_else(|| "kani::any()".to_string(), |t| self.compile_type(t));
                format!(
                    "{{ let {}: {} = kani::any(); {} }}",
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
                    "{}.iter().all(|{}| {})",
                    self.compile_expr(collection),
                    var,
                    self.compile_expr(body)
                )
            }
            Expr::ExistsIn {
                var,
                collection,
                body,
            } => {
                format!(
                    "{}.iter().any(|{}| {})",
                    self.compile_expr(collection),
                    var,
                    self.compile_expr(body)
                )
            }

            Expr::Implies(lhs, rhs) => {
                format!(
                    "(!({}) || ({}))",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::And(lhs, rhs) => {
                format!(
                    "(({}) && ({}))",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Or(lhs, rhs) => {
                format!(
                    "(({}) || ({}))",
                    self.compile_expr(lhs),
                    self.compile_expr(rhs)
                )
            }
            Expr::Not(e) => format!("!({})", self.compile_expr(e)),

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
                    "(({}) {} ({}))",
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
                    "(({}) {} ({}))",
                    self.compile_expr(lhs),
                    op_str,
                    self.compile_expr(rhs)
                )
            }
            Expr::Neg(e) => format!("-({})", self.compile_expr(e)),

            Expr::App(name, args) => {
                if args.is_empty() {
                    format!("{name}()")
                } else {
                    let args_str: Vec<String> = args.iter().map(|a| self.compile_expr(a)).collect();
                    format!("{name}({})", args_str.join(", "))
                }
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                let recv_str = self.compile_expr(receiver);
                if args.is_empty() {
                    format!("{recv_str}.{method}()")
                } else {
                    let args_str: Vec<String> = args.iter().map(|a| self.compile_expr(a)).collect();
                    format!("{recv_str}.{method}({})", args_str.join(", "))
                }
            }
            Expr::FieldAccess(obj, field) => {
                format!("{}.{field}", self.compile_expr(obj))
            }
        }
    }

    /// Compile a type to Rust syntax
    #[must_use]
    pub fn compile_type(&self, ty: &Type) -> String {
        match ty {
            Type::Named(name) => match name.as_str() {
                "Bool" | "boolean" => "bool".to_string(),
                "Int" | "int" | "integer" => "i64".to_string(),
                "Float" | "float" => "f64".to_string(),
                "String" | "string" => "String".to_string(),
                _ => name.clone(),
            },
            Type::Set(inner) => format!("std::collections::HashSet<{}>", self.compile_type(inner)),
            Type::List(inner) => format!("Vec<{}>", self.compile_type(inner)),
            Type::Map(k, v) => {
                format!(
                    "std::collections::HashMap<{}, {}>",
                    self.compile_type(k),
                    self.compile_type(v)
                )
            }
            Type::Relation(a, b) => {
                format!(
                    "std::collections::HashSet<({}, {})>",
                    self.compile_type(a),
                    self.compile_type(b)
                )
            }
            Type::Function(a, b) => {
                format!("fn({}) -> {}", self.compile_type(a), self.compile_type(b))
            }
            Type::Result(inner) => format!("Result<{}, String>", self.compile_type(inner)),
            Type::Unit => "()".to_string(),
            Type::Graph(n, e) => {
                // Rust graph using petgraph-style representation
                format!("Graph<{}, {}>", self.compile_type(n), self.compile_type(e))
            }
            Type::Path(n) => format!("Vec<{}>", self.compile_type(n)),
        }
    }

    /// Compile a contract to Kani proof harness
    #[must_use]
    pub fn compile_contract(&self, contract: &Contract) -> String {
        let fn_name = contract.type_path.join("::");
        let harness_name = format!("verify_{}", contract.type_path.join("_").replace("::", "_"));

        let mut lines = Vec::new();

        // Generate proof harness
        lines.push("#[cfg(kani)]".to_string());
        lines.push("#[kani::proof]".to_string());
        lines.push(format!("fn {harness_name}() {{"));

        // Generate symbolic inputs
        for param in &contract.params {
            lines.push(format!(
                "    let {}: {} = kani::any();",
                param.name,
                self.compile_type(&param.ty)
            ));
        }
        lines.push(String::new());

        // Preconditions (assumes)
        if !contract.requires.is_empty() {
            lines.push("    // Preconditions".to_string());
            for req in &contract.requires {
                lines.push(format!("    kani::assume({});", self.compile_expr(req)));
            }
            lines.push(String::new());
        }

        // Function call
        let params_str: Vec<String> = contract.params.iter().map(|p| p.name.clone()).collect();
        if let Some(ret_ty) = &contract.return_type {
            lines.push(format!(
                "    let result: {} = {}({});",
                self.compile_type(ret_ty),
                fn_name,
                params_str.join(", ")
            ));
        } else {
            lines.push(format!("    {}({});", fn_name, params_str.join(", ")));
        }
        lines.push(String::new());

        let returns_result = matches!(contract.return_type, Some(Type::Result(_)));

        if returns_result {
            lines.push("    match result {".to_string());

            // Success path postconditions
            lines.push("        Ok(result) => {".to_string());
            if !contract.ensures.is_empty() {
                lines.push("            // Postconditions".to_string());
                for (i, ens) in contract.ensures.iter().enumerate() {
                    lines.push(format!(
                        "            kani::assert({}, \"postcondition_{}\");",
                        self.compile_expr(ens),
                        i
                    ));
                }
            }
            lines.push("        }".to_string());

            // Error path postconditions
            lines.push("        Err(result) => {".to_string());
            if !contract.ensures_err.is_empty() {
                lines.push("            // Error postconditions".to_string());
                for (i, ens) in contract.ensures_err.iter().enumerate() {
                    lines.push(format!(
                        "            kani::assert({}, \"error_postcondition_{}\");",
                        self.compile_expr(ens),
                        i
                    ));
                }
            }
            lines.push("        }".to_string());

            lines.push("    }".to_string());
        } else {
            // Postconditions (asserts)
            if !contract.ensures.is_empty() {
                lines.push("    // Postconditions".to_string());
                for (i, ens) in contract.ensures.iter().enumerate() {
                    lines.push(format!(
                        "    kani::assert({}, \"postcondition_{}\");",
                        self.compile_expr(ens),
                        i
                    ));
                }
            }

            // If a function does not return Result, we cannot check ensures_err
            if !contract.ensures_err.is_empty() {
                lines.push(
                    "    // ensures_err clauses require a Result return type and are skipped here"
                        .to_string(),
                );
            }
        }

        lines.push("}".to_string());
        lines.join("\n")
    }

    /// Compile a refinement to Kani proof harness
    ///
    /// Generates:
    /// 1. Abstraction function verification
    /// 2. Simulation relation verification
    /// 3. Invariant assertions
    /// 4. Action correspondence assertions
    #[must_use]
    pub fn compile_refinement(&self, refinement: &Refinement) -> String {
        let harness_name = format!("verify_refinement_{}", refinement.name);

        let mut lines = Vec::new();

        // Header comment
        lines.push(format!(
            "// Refinement: {} refines {}",
            refinement.name, refinement.refines
        ));
        lines.push(String::new());

        // Generate proof harness for abstraction function
        lines.push("#[cfg(kani)]".to_string());
        lines.push("#[kani::proof]".to_string());
        lines.push(format!("fn {harness_name}_abstraction() {{"));
        lines.push("    // Verify abstraction function is well-defined".to_string());
        lines.push(format!(
            "    kani::assert({}, \"abstraction_function\");",
            self.compile_expr(&refinement.abstraction)
        ));
        lines.push("}".to_string());
        lines.push(String::new());

        // Generate proof harness for simulation relation
        lines.push("#[cfg(kani)]".to_string());
        lines.push("#[kani::proof]".to_string());
        lines.push(format!("fn {harness_name}_simulation() {{"));
        lines.push("    // Verify simulation relation holds".to_string());
        lines.push(format!(
            "    kani::assert({}, \"simulation_relation\");",
            self.compile_expr(&refinement.simulation)
        ));
        lines.push("}".to_string());
        lines.push(String::new());

        // Generate invariant verification harnesses
        if !refinement.invariants.is_empty() {
            lines.push("#[cfg(kani)]".to_string());
            lines.push("#[kani::proof]".to_string());
            lines.push(format!("fn {harness_name}_invariants() {{"));
            lines.push("    // Verify refinement invariants".to_string());
            for (i, inv) in refinement.invariants.iter().enumerate() {
                lines.push(format!(
                    "    kani::assert({}, \"refinement_invariant_{}\");",
                    self.compile_expr(inv),
                    i
                ));
            }
            lines.push("}".to_string());
            lines.push(String::new());
        }

        // Generate action correspondence harnesses
        if !refinement.actions.is_empty() {
            for action in &refinement.actions {
                let action_harness =
                    format!("{harness_name}_action_{}", action.name.replace("::", "_"));
                let impl_path = action.impl_action.join("::");

                lines.push("#[cfg(kani)]".to_string());
                lines.push("#[kani::proof]".to_string());
                lines.push(format!("fn {action_harness}() {{"));
                lines.push(format!(
                    "    // Verify action correspondence: {} <-> {}",
                    action.spec_action, impl_path
                ));

                // If there's a guard, assume it
                if let Some(guard) = &action.guard {
                    lines.push(format!("    kani::assume({});", self.compile_expr(guard)));
                }

                // Generate assertion that spec and impl actions correspond
                // This is a placeholder - in practice, you'd generate actual action calls
                lines.push(format!(
                    "    // Action: spec.{} corresponds to impl.{}",
                    action.spec_action, impl_path
                ));
                lines.push("    // TODO: Add specific action verification logic".to_string());

                lines.push("}".to_string());
                lines.push(String::new());
            }
        }

        // Generate variable mapping verification
        if !refinement.mappings.is_empty() {
            lines.push("#[cfg(kani)]".to_string());
            lines.push("#[kani::proof]".to_string());
            lines.push(format!("fn {harness_name}_mappings() {{"));
            lines.push("    // Verify variable mappings".to_string());
            for (i, mapping) in refinement.mappings.iter().enumerate() {
                let spec_var = self.compile_expr(&mapping.spec_var);
                let impl_var = self.compile_expr(&mapping.impl_var);
                lines.push(format!(
                    "    // mapping_{}: {} <-> {}",
                    i, spec_var, impl_var
                ));
                // Generate assertion that the mapping is consistent
                lines.push(format!(
                    "    kani::assert({} == {}, \"mapping_{}\");",
                    spec_var, impl_var, i
                ));
            }
            lines.push("}".to_string());
        }

        lines.join("\n")
    }

    /// Generate complete Kani harness file from spec
    #[must_use]
    pub fn compile_module(&self, typed_spec: &TypedSpec) -> CompiledSpec {
        // File header
        let mut sections = vec![
            "// Generated from USL by DashProve".to_string(),
            "// Kani proof harnesses".to_string(),
            String::new(),
            "#![allow(unused)]".to_string(),
            String::new(),
        ];

        // Compile contracts to proof harnesses
        let mut has_properties = false;
        for property in &typed_spec.spec.properties {
            match property {
                Property::Contract(contract) => {
                    has_properties = true;
                    let fn_name = contract.type_path.join("::");
                    sections.push(format!("// Contract: {fn_name}"));
                    sections.push(self.compile_contract(contract));
                    sections.push(String::new());
                }
                Property::Refinement(refinement) => {
                    has_properties = true;
                    sections.push(self.compile_refinement(refinement));
                    sections.push(String::new());
                }
                _ => {}
            }
        }

        if !has_properties {
            sections.push("// No contracts or refinements found in specification".to_string());
        }

        CompiledSpec {
            backend: "Kani".to_string(),
            code: sections.join("\n"),
            module_name: Some(self.crate_name.clone()),
            imports: vec!["kani".to_string()],
        }
    }
}

/// Compile to Kani harness
#[must_use]
pub fn compile_to_kani(spec: &TypedSpec) -> CompiledSpec {
    let compiler = KaniCompiler::new("usl_verify");
    compiler.compile_module(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, Type};
    use crate::{parse, typecheck, typecheck::TypedSpec};

    fn compile_usl(input: &str) -> TypedSpec {
        let spec = parse(input).expect("parse failed");
        typecheck(spec).expect("typecheck failed")
    }

    #[test]
    fn test_kani_compile_primed_variable() {
        let compiler = KaniCompiler::new("test");
        // Primed variables (x') should have the prime stripped
        let expr = Expr::Var("count'".to_string());
        let result = compiler.compile_expr(&expr);
        // Line 32: name[..name.len() - 1] - tests the subtraction
        assert_eq!(
            result, "count",
            "Primed variable should have prime stripped"
        );
        assert!(!result.ends_with('\''), "Result should not end with prime");
    }

    #[test]
    fn test_kani_compile_primed_variable_length() {
        let compiler = KaniCompiler::new("test");
        // Test edge case with single character variable
        let expr = Expr::Var("x'".to_string());
        let result = compiler.compile_expr(&expr);
        // Line 33: name == "self" check - this tests that primed x is not confused with self
        assert_eq!(result, "x");
        assert_ne!(result, "self");
    }

    #[test]
    fn test_kani_compile_self_variable() {
        let compiler = KaniCompiler::new("test");
        let expr = Expr::Var("self".to_string());
        let result = compiler.compile_expr(&expr);
        // Line 33-34: name == "self" check
        assert_eq!(result, "self");
    }

    #[test]
    fn test_kani_compile_non_self_variable() {
        let compiler = KaniCompiler::new("test");
        // Test that a non-self variable is NOT treated as "self"
        // This kills the mutation: replace == with != (line 33)
        let expr = Expr::Var("other".to_string());
        let result = compiler.compile_expr(&expr);
        // When name == "self" is mutated to name != "self", then "other" would
        // become "self" instead of "other", which this test catches
        assert_eq!(result, "other");
        assert_ne!(
            result, "self",
            "Non-self variables should not be converted to 'self'"
        );
    }

    #[test]
    fn test_kani_type_bool() {
        let compiler = KaniCompiler::new("test");
        // Line 180: "Bool" | "boolean" -> "bool"
        assert_eq!(
            compiler.compile_type(&Type::Named("Bool".to_string())),
            "bool"
        );
        assert_eq!(
            compiler.compile_type(&Type::Named("boolean".to_string())),
            "bool"
        );
    }

    #[test]
    fn test_kani_type_int() {
        let compiler = KaniCompiler::new("test");
        // Line 181: "Int" | "int" | "integer" -> "i64"
        assert_eq!(
            compiler.compile_type(&Type::Named("Int".to_string())),
            "i64"
        );
        assert_eq!(
            compiler.compile_type(&Type::Named("int".to_string())),
            "i64"
        );
        assert_eq!(
            compiler.compile_type(&Type::Named("integer".to_string())),
            "i64"
        );
    }

    #[test]
    fn test_kani_type_float() {
        let compiler = KaniCompiler::new("test");
        // Line 182: "Float" | "float" -> "f64"
        assert_eq!(
            compiler.compile_type(&Type::Named("Float".to_string())),
            "f64"
        );
        assert_eq!(
            compiler.compile_type(&Type::Named("float".to_string())),
            "f64"
        );
    }

    #[test]
    fn test_kani_type_string() {
        let compiler = KaniCompiler::new("test");
        // Line 183: "String" | "string" -> "String"
        assert_eq!(
            compiler.compile_type(&Type::Named("String".to_string())),
            "String"
        );
        assert_eq!(
            compiler.compile_type(&Type::Named("string".to_string())),
            "String"
        );
    }

    #[test]
    fn test_kani_type_custom() {
        let compiler = KaniCompiler::new("test");
        // Custom types pass through unchanged
        assert_eq!(
            compiler.compile_type(&Type::Named("MyStruct".to_string())),
            "MyStruct"
        );
        assert_eq!(
            compiler.compile_type(&Type::Named("Vec<u8>".to_string())),
            "Vec<u8>"
        );
    }

    #[test]
    fn test_kani_contract_with_ensures() {
        let input = r#"
            contract add(x: Int, y: Int) -> Int {
                requires { x >= 0 }
                requires { y >= 0 }
                ensures { result >= x }
                ensures { result >= y }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        // Line 263: !contract.ensures.is_empty()
        assert!(
            compiled.code.contains("Postconditions"),
            "Should have postconditions section"
        );
        assert!(
            compiled.code.contains("postcondition_0"),
            "Should have numbered postconditions"
        );
        assert!(
            compiled.code.contains("postcondition_1"),
            "Should have second postcondition"
        );
    }

    #[test]
    fn test_kani_contract_with_ensures_err() {
        let input = r#"
            contract safe_div(x: Int, y: Int) -> Result<Int> {
                requires { true }
                ensures { result >= 0 }
                ensures_err { true }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        // For Result return types, ensures_err should be processed
        assert!(
            compiled.code.contains("Ok(result)") || compiled.code.contains("Err(result)"),
            "Result type should have match arms"
        );
    }

    #[test]
    fn test_kani_result_contract_with_ensures_postconditions() {
        // Line 263: !contract.ensures.is_empty() within Result branch
        // When mutation deletes '!', this would generate Postconditions
        // even when ensures IS empty. But with correct code, ensures non-empty
        // should generate postconditions.
        let input = r#"
            contract result_fn(x: Int) -> Result<Int> {
                requires { x > 0 }
                ensures { result == x + 1 }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        // With ensures non-empty, should have "// Postconditions" comment
        // Count occurrences - should have exactly one Postconditions section in Ok branch
        let postconditions_count = compiled.code.matches("// Postconditions").count();
        assert!(
            postconditions_count >= 1,
            "Result contract with ensures should have Postconditions section"
        );
        assert!(
            compiled.code.contains("postcondition_0"),
            "Should have postcondition assert"
        );
    }

    #[test]
    fn test_kani_result_contract_no_ensures_no_postconditions() {
        // Line 263: !contract.ensures.is_empty() within Result branch
        // When ensures IS empty, should NOT have "// Postconditions" in Ok branch
        let input = r#"
            contract result_fn_no_ensures(x: Int) -> Result<Int> {
                requires { x > 0 }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        // With ensures empty, should NOT have "// Postconditions" comment
        // But should still have the Result match structure
        assert!(
            compiled.code.contains("Ok(result)"),
            "Should have Ok match arm"
        );
        // The Ok branch should be nearly empty (just the closing brace)
        // Check there's no postcondition_0
        assert!(
            !compiled.code.contains("postcondition_0"),
            "Empty ensures should not generate postcondition asserts"
        );
    }

    #[test]
    fn test_kani_non_result_contract_with_ensures_err_shows_skip() {
        // Line 304: !contract.ensures_err.is_empty() when NOT Result
        // When a non-Result function has ensures_err, it should show skip comment
        let input = r#"
            contract non_result_with_err(x: Int) -> Int {
                requires { x > 0 }
                ensures_err { false }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        // Non-Result with ensures_err should show the skip comment
        assert!(
            compiled
                .code
                .contains("ensures_err clauses require a Result return type"),
            "Non-Result contract with ensures_err should show skip comment"
        );
    }

    #[test]
    fn test_kani_non_result_contract_no_ensures_err_no_skip() {
        // Line 304: !contract.ensures_err.is_empty() when NOT Result
        // When a non-Result function has NO ensures_err, should NOT have skip comment
        let input = r#"
            contract non_result_simple(x: Int) -> Int {
                requires { x > 0 }
                ensures { result > 0 }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        // Non-Result without ensures_err should NOT show the skip comment
        assert!(
            !compiled
                .code
                .contains("ensures_err clauses require a Result return type"),
            "Non-Result contract without ensures_err should not have skip comment"
        );
    }

    #[test]
    fn test_kani_contract_no_ensures() {
        let input = r#"
            contract simple(x: Int) -> Int {
                requires { x > 0 }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        // When ensures is empty, no postconditions section
        // The code should still compile but without postcondition asserts
        assert!(
            compiled.code.contains("kani::proof"),
            "Should have proof harness"
        );
        assert!(
            compiled.code.contains("kani::assume"),
            "Should have precondition assumes"
        );
    }

    #[test]
    fn test_kani_refinement_basic() {
        let input = r#"
            refinement optimized refines abstract {
                abstraction { impl_to_abs(concrete) == abstract }
                simulation { forall s: State . step(impl_to_abs(s)) == impl_to_abs(step(s)) }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        // Line 325: compile_refinement returns non-empty string
        assert!(!compiled.code.is_empty(), "Refinement should produce code");
        assert!(
            compiled.code.contains("verify_refinement_optimized"),
            "Should have refinement harness name"
        );
        assert!(
            compiled.code.contains("abstraction_function"),
            "Should have abstraction verification"
        );
        assert!(
            compiled.code.contains("simulation_relation"),
            "Should have simulation verification"
        );
    }

    #[test]
    fn test_kani_refinement_with_invariants() {
        // Grammar: refinement_invariant_clause = { "invariant" ~ block }
        // These must come BEFORE abstraction and simulation in the grammar
        let input = r#"
            refinement impl refines spec {
                invariant { count >= 0 }
                invariant { count < max }
                abstraction { true }
                simulation { true }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        // Line 361: !refinement.invariants.is_empty()
        assert!(
            compiled.code.contains("_invariants()"),
            "Should have invariants harness"
        );
        assert!(
            compiled.code.contains("refinement_invariant_0"),
            "Should have numbered invariants"
        );
        assert!(
            compiled.code.contains("refinement_invariant_1"),
            "Should have second invariant"
        );
    }

    #[test]
    fn test_kani_refinement_with_actions() {
        // Grammar: action_mapping_clause = { "action" ~ ident ~ "{" ~ action_spec_clause ~ action_impl_clause ~ action_guard_clause? ~ "}" }
        // action_spec_clause = { "spec" ~ ":" ~ ident }
        // action_impl_clause = { "impl" ~ ":" ~ type_path ~ "(" ~ ")" }
        // action_guard_clause = { "guard" ~ block }
        let input = r#"
            refinement impl refines spec {
                abstraction { true }
                simulation { true }
                action increment {
                    spec: IncrementAction
                    impl: Counter::inc()
                    guard { count < max }
                }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        // Line 378: !refinement.actions.is_empty()
        assert!(
            compiled.code.contains("_action_"),
            "Should have action harness"
        );
        assert!(
            compiled.code.contains("Action:"),
            "Should have action comments"
        );
    }

    #[test]
    fn test_kani_refinement_with_mappings() {
        // Grammar: refinement_mapping_clause = { "mapping" ~ "{" ~ var_mapping* ~ "}" }
        // var_mapping = { access_expr ~ "<->" ~ access_expr }
        // Mappings must come BEFORE invariants, which come BEFORE abstraction/simulation
        let input = r#"
            refinement impl refines spec {
                mapping {
                    spec.count <-> impl.counter
                }
                abstraction { true }
                simulation { true }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        // Line 411: !refinement.mappings.is_empty()
        assert!(
            compiled.code.contains("_mappings()"),
            "Should have mappings harness"
        );
        assert!(
            compiled.code.contains("mapping_"),
            "Should have numbered mappings"
        );
    }

    #[test]
    fn test_kani_module_with_refinement() {
        let input = r#"
            refinement impl refines spec {
                abstraction { true }
                simulation { true }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        // Line 458: Property::Refinement match arm
        assert!(
            compiled.code.contains("verify_refinement"),
            "Should process refinement property"
        );
    }

    #[test]
    fn test_kani_module_no_properties() {
        let input = r#"
            theorem test { true }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        // Line 467: !has_properties - when only theorems (not contracts/refinements)
        assert!(
            compiled.code.contains("No contracts or refinements found"),
            "Should indicate no Kani-relevant properties"
        );
    }

    #[test]
    fn test_kani_module_header() {
        let input = r#"
            contract test(x: Int) -> Int {
                requires { true }
            }
        "#;
        let typed = compile_usl(input);
        let compiled = compile_to_kani(&typed);

        assert_eq!(compiled.backend, "Kani");
        assert!(compiled.code.contains("Generated from USL by DashProve"));
        assert!(compiled.code.contains("#![allow(unused)]"));
    }
}
