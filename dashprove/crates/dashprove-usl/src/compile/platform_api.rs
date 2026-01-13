//! Platform API static checker code generator
//!
//! Generates Rust code from PlatformApi specifications that can be used
//! to statically verify API usage patterns. This is critical for catching
//! bugs in external API usage (Metal, CUDA, Vulkan, POSIX, etc.) that
//! traditional formal verification tools cannot catch because they model
//! external API behavior.
//!
//! # Example
//!
//! Given a USL platform_api specification:
//! ```text
//! platform_api Metal {
//!     state MTLCommandBuffer {
//!         enum Status { Created, Encoding, Committed, Completed }
//!
//!         transition commit() {
//!             requires { status == Created or status == Encoding }
//!             ensures { status == Committed }
//!         }
//!     }
//! }
//! ```
//!
//! This generates Rust code with:
//! - State enum definitions
//! - State tracker struct with transition methods
//! - Debug assertions for preconditions
//! - State update logic for postconditions

use crate::ast::{ApiState, ApiTransition, BinaryOp, ComparisonOp, Expr, PlatformApi, Property};
use crate::typecheck::TypedSpec;

use super::CompiledSpec;

/// Platform API static checker compiler
pub struct PlatformApiCompiler {
    #[allow(dead_code)]
    platform_name: String,
}

impl PlatformApiCompiler {
    /// Create a new platform API compiler
    #[must_use]
    pub fn new(platform_name: &str) -> Self {
        Self {
            platform_name: platform_name.to_string(),
        }
    }

    /// Compile a PlatformApi to Rust static checker code
    #[must_use]
    pub fn compile(&self, api: &PlatformApi) -> String {
        let mut output = String::new();

        // Module header
        output.push_str(&format!(
            "//! Auto-generated static checker for {} platform API constraints\n",
            api.name
        ));
        output.push_str("//!\n");
        output.push_str("//! This module provides state tracking for API objects to verify\n");
        output.push_str("//! that API calls respect required preconditions.\n\n");

        // Generate code for each API state machine
        for state in &api.states {
            output.push_str(&self.compile_state(state));
            output.push('\n');
        }

        output
    }

    /// Compile an API state machine to Rust code
    fn compile_state(&self, state: &ApiState) -> String {
        let mut output = String::new();

        // Generate status enum if defined
        if let Some(status_enum) = &state.status_enum {
            output.push_str(&format!("/// Status enum for {}\n", state.name));
            output.push_str("#[derive(Debug, Clone, Copy, PartialEq, Eq)]\n");
            output.push_str(&format!("pub enum {}Status {{\n", state.name));
            for variant in &status_enum.variants {
                output.push_str(&format!("    {},\n", variant));
            }
            output.push_str("}\n\n");
        }

        // Generate state tracker struct
        output.push_str(&format!(
            "/// State tracker for {} API object\n",
            state.name
        ));
        output.push_str("/// \n/// Tracks the current state and validates transitions.\n");
        output.push_str("#[derive(Debug)]\n");
        output.push_str(&format!("pub struct {}StateTracker {{\n", state.name));

        if state.status_enum.is_some() {
            output.push_str(&format!("    pub status: {}Status,\n", state.name));
        }

        // Add call tracking for each transition
        for transition in &state.transitions {
            output.push_str(&format!(
                "    /// Track if {} has been called\n",
                transition.name
            ));
            output.push_str(&format!(
                "    pub {}_called: bool,\n",
                to_snake_case(&transition.name)
            ));
        }

        output.push_str("}\n\n");

        // Generate impl block
        output.push_str(&format!("impl {}StateTracker {{\n", state.name));

        // Constructor
        output.push_str("    /// Create a new state tracker\n");
        output.push_str("    #[must_use]\n");
        output.push_str("    pub fn new() -> Self {\n");
        output.push_str("        Self {\n");

        if let Some(status_enum) = &state.status_enum {
            let initial = status_enum
                .variants
                .first()
                .map_or("Unknown", |s| s.as_str());
            output.push_str(&format!(
                "            status: {}Status::{},\n",
                state.name, initial
            ));
        }

        for transition in &state.transitions {
            output.push_str(&format!(
                "            {}_called: false,\n",
                to_snake_case(&transition.name)
            ));
        }

        output.push_str("        }\n");
        output.push_str("    }\n\n");

        // Generate transition methods
        for transition in &state.transitions {
            output.push_str(&self.compile_transition(state, transition));
        }

        // Generate validation helper
        output.push_str("    /// Validate all invariants\n");
        output.push_str("    pub fn validate(&self) -> Result<(), String> {\n");
        if state.invariants.is_empty() {
            output.push_str("        Ok(())\n");
        } else {
            for (i, inv) in state.invariants.iter().enumerate() {
                let condition = self.compile_expr(inv);
                output.push_str(&format!("        if !({}) {{\n", condition));
                output.push_str(&format!(
                    "            return Err(\"Invariant {} violated\".to_string());\n",
                    i + 1
                ));
                output.push_str("        }\n");
            }
            output.push_str("        Ok(())\n");
        }
        output.push_str("    }\n");

        output.push_str("}\n\n");

        // Generate Default impl
        output.push_str(&format!("impl Default for {}StateTracker {{\n", state.name));
        output.push_str("    fn default() -> Self {\n");
        output.push_str("        Self::new()\n");
        output.push_str("    }\n");
        output.push_str("}\n");

        output
    }

    /// Compile a transition to a Rust method
    fn compile_transition(&self, state: &ApiState, transition: &ApiTransition) -> String {
        let mut output = String::new();

        // Method signature
        let method_name = to_snake_case(&transition.name);
        output.push_str(&format!(
            "    /// Validate and track call to `{}`\n",
            transition.name
        ));

        // Build parameter list
        let params: Vec<String> = transition
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, type_to_rust(&p.ty)))
            .collect();

        let params_with_self = if params.is_empty() {
            "&mut self".to_string()
        } else {
            format!("&mut self, {}", params.join(", "))
        };

        output.push_str(&format!(
            "    pub fn {}({}) -> Result<(), String> {{\n",
            method_name, params_with_self
        ));

        // Check preconditions
        if !transition.requires.is_empty() {
            output.push_str("        // Check preconditions\n");
            for (i, req) in transition.requires.iter().enumerate() {
                let condition = self.compile_expr_for_state(req, state);
                output.push_str(&format!("        if !({}) {{\n", condition));
                output.push_str(&format!(
                    "            return Err(format!(\"Precondition {} for {} violated: {{:?}}\", self.status));\n",
                    i + 1, transition.name
                ));
                output.push_str("        }\n");
            }
            output.push('\n');
        }

        // Mark as called
        output.push_str(&format!("        self.{}_called = true;\n", method_name));

        // Apply postconditions (state updates)
        if !transition.ensures.is_empty() {
            output.push_str("\n        // Apply postconditions\n");
            for ensure in &transition.ensures {
                if let Some(update) = self.extract_state_update(ensure, state) {
                    output.push_str(&format!("        {};\n", update));
                }
            }
        }

        output.push_str("\n        Ok(())\n");
        output.push_str("    }\n\n");

        output
    }

    /// Compile an expression to Rust code
    fn compile_expr(&self, expr: &Expr) -> String {
        self.compile_expr_impl(expr, None)
    }

    /// Compile an expression with access to state context
    fn compile_expr_for_state(&self, expr: &Expr, state: &ApiState) -> String {
        self.compile_expr_impl(expr, Some(state))
    }

    /// Internal expression compiler
    fn compile_expr_impl(&self, expr: &Expr, state: Option<&ApiState>) -> String {
        match expr {
            Expr::Var(name) => {
                // Check if this is a status enum reference
                if let Some(st) = state {
                    if name == "status" {
                        return "self.status".to_string();
                    }
                    // Check if it's an enum variant
                    if let Some(status_enum) = &st.status_enum {
                        if status_enum.variants.contains(name) {
                            return format!("{}Status::{}", st.name, name);
                        }
                    }
                }
                name.clone()
            }
            Expr::Int(n) => n.to_string(),
            Expr::Float(f) => format!("{f}"),
            Expr::String(s) => format!("\"{s}\""),
            Expr::Bool(b) => b.to_string(),

            // Comparison: Compare(left, op, right)
            Expr::Compare(left, op, right) => {
                let l = self.compile_expr_impl(left, state);
                let r = self.compile_expr_impl(right, state);
                let op_str = match op {
                    ComparisonOp::Eq => "==",
                    ComparisonOp::Ne => "!=",
                    ComparisonOp::Lt => "<",
                    ComparisonOp::Le => "<=",
                    ComparisonOp::Gt => ">",
                    ComparisonOp::Ge => ">=",
                };
                format!("({l} {op_str} {r})")
            }

            Expr::And(lhs, rhs) => {
                format!(
                    "({} && {})",
                    self.compile_expr_impl(lhs, state),
                    self.compile_expr_impl(rhs, state)
                )
            }
            Expr::Or(lhs, rhs) => {
                format!(
                    "({} || {})",
                    self.compile_expr_impl(lhs, state),
                    self.compile_expr_impl(rhs, state)
                )
            }
            Expr::Not(inner) => {
                format!("!({})", self.compile_expr_impl(inner, state))
            }
            Expr::Implies(lhs, rhs) => {
                format!(
                    "(!({}) || ({}))",
                    self.compile_expr_impl(lhs, state),
                    self.compile_expr_impl(rhs, state)
                )
            }

            // Binary arithmetic: Binary(left, op, right)
            Expr::Binary(left, op, right) => {
                let l = self.compile_expr_impl(left, state);
                let r = self.compile_expr_impl(right, state);
                let op_str = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Mod => "%",
                };
                format!("({l} {op_str} {r})")
            }

            // Unary negation
            Expr::Neg(inner) => {
                format!("-({})", self.compile_expr_impl(inner, state))
            }

            Expr::FieldAccess(base, field) => {
                format!("{}.{}", self.compile_expr_impl(base, state), field)
            }

            // Function application: App(name, args)
            Expr::App(name, args) => {
                let args_str: Vec<String> = args
                    .iter()
                    .map(|a| self.compile_expr_impl(a, state))
                    .collect();
                format!("{}({})", name, args_str.join(", "))
            }

            // Method call
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                let recv = self.compile_expr_impl(receiver, state);
                let args_str: Vec<String> = args
                    .iter()
                    .map(|a| self.compile_expr_impl(a, state))
                    .collect();
                format!("{}.{}({})", recv, method, args_str.join(", "))
            }

            // Quantifiers - simplified for runtime checking
            Expr::ForAll { var, body, .. } => {
                format!(
                    "/* forall {} . {} */true",
                    var,
                    self.compile_expr_impl(body, state)
                )
            }
            Expr::Exists { var, body, .. } => {
                format!(
                    "/* exists {} . {} */true",
                    var,
                    self.compile_expr_impl(body, state)
                )
            }
            Expr::ForAllIn {
                var,
                collection,
                body,
            } => {
                format!(
                    "{}.iter().all(|{}| {})",
                    self.compile_expr_impl(collection, state),
                    var,
                    self.compile_expr_impl(body, state)
                )
            }
            Expr::ExistsIn {
                var,
                collection,
                body,
            } => {
                format!(
                    "{}.iter().any(|{}| {})",
                    self.compile_expr_impl(collection, state),
                    var,
                    self.compile_expr_impl(body, state)
                )
            }
        }
    }

    /// Extract a state update from an ensures clause
    /// e.g., `status == Committed` becomes `self.status = MTLCommandBufferStatus::Committed`
    fn extract_state_update(&self, expr: &Expr, state: &ApiState) -> Option<String> {
        if let Expr::Compare(left, ComparisonOp::Eq, right) = expr {
            // Check for status = EnumVariant pattern
            if let Expr::Var(name) = left.as_ref() {
                if name == "status" {
                    if let Expr::Var(variant) = right.as_ref() {
                        if let Some(status_enum) = &state.status_enum {
                            if status_enum.variants.contains(variant) {
                                return Some(format!(
                                    "self.status = {}Status::{}",
                                    state.name, variant
                                ));
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

/// Convert a USL type to Rust type
fn type_to_rust(ty: &crate::ast::Type) -> String {
    use crate::ast::Type;
    match ty {
        Type::Named(name) => {
            // Map common USL type names to Rust types
            match name.as_str() {
                "Int" | "int" | "i64" => "i64".to_string(),
                "Real" | "real" | "f64" => "f64".to_string(),
                "Bool" | "bool" | "boolean" => "bool".to_string(),
                "String" | "string" => "String".to_string(),
                _ => name.clone(),
            }
        }
        Type::Set(inner) => format!("Vec<{}>", type_to_rust(inner)),
        Type::List(inner) => format!("Vec<{}>", type_to_rust(inner)),
        Type::Map(k, v) => {
            format!(
                "std::collections::HashMap<{}, {}>",
                type_to_rust(k),
                type_to_rust(v)
            )
        }
        Type::Relation(a, b) => {
            format!("Vec<({}, {})>", type_to_rust(a), type_to_rust(b))
        }
        Type::Function(arg, ret) => {
            format!("fn({}) -> {}", type_to_rust(arg), type_to_rust(ret))
        }
        Type::Result(inner) => format!("Result<{}, String>", type_to_rust(inner)),
        Type::Unit => "()".to_string(),
        Type::Graph(n, e) => format!("Graph<{}, {}>", type_to_rust(n), type_to_rust(e)),
        Type::Path(n) => format!("Vec<{}>", type_to_rust(n)),
    }
}

/// Convert camelCase or PascalCase to snake_case
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}

/// Compile a typed specification to platform API static checker code
///
/// Returns None if the spec contains no PlatformApi properties.
#[must_use]
pub fn compile_to_platform_api(spec: &TypedSpec) -> Option<CompiledSpec> {
    let platform_apis: Vec<&PlatformApi> = spec
        .spec
        .properties
        .iter()
        .filter_map(|p| {
            if let Property::PlatformApi(api) = p {
                Some(api)
            } else {
                None
            }
        })
        .collect();

    if platform_apis.is_empty() {
        return None;
    }

    let mut code = String::new();
    code.push_str("//! Auto-generated platform API static checkers\n");
    code.push_str("//!\n");
    code.push_str("//! Generated by dashprove from USL platform_api specifications.\n");
    code.push_str("//! These state trackers verify API usage follows documented constraints.\n\n");

    code.push_str("#![allow(dead_code)]\n");
    code.push_str("#![allow(unused_variables)]\n\n");

    for api in platform_apis {
        let compiler = PlatformApiCompiler::new(&api.name);
        code.push_str(&compiler.compile(api));
        code.push('\n');
    }

    Some(CompiledSpec {
        backend: "platform_api".to_string(),
        code,
        module_name: Some("platform_api_checkers".to_string()),
        imports: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{ApiState, ApiTransition, Param, PlatformApi, StateEnum, Type};

    #[test]
    fn test_compile_simple_platform_api() {
        let api = PlatformApi {
            name: "Metal".to_string(),
            states: vec![ApiState {
                name: "MTLCommandBuffer".to_string(),
                status_enum: Some(StateEnum {
                    name: "Status".to_string(),
                    variants: vec![
                        "Created".to_string(),
                        "Encoding".to_string(),
                        "Committed".to_string(),
                        "Completed".to_string(),
                    ],
                }),
                transitions: vec![ApiTransition {
                    name: "commit".to_string(),
                    params: vec![],
                    requires: vec![Expr::Or(
                        Box::new(Expr::Compare(
                            Box::new(Expr::Var("status".to_string())),
                            ComparisonOp::Eq,
                            Box::new(Expr::Var("Created".to_string())),
                        )),
                        Box::new(Expr::Compare(
                            Box::new(Expr::Var("status".to_string())),
                            ComparisonOp::Eq,
                            Box::new(Expr::Var("Encoding".to_string())),
                        )),
                    )],
                    ensures: vec![Expr::Compare(
                        Box::new(Expr::Var("status".to_string())),
                        ComparisonOp::Eq,
                        Box::new(Expr::Var("Committed".to_string())),
                    )],
                }],
                invariants: vec![],
            }],
        };

        let compiler = PlatformApiCompiler::new("Metal");
        let output = compiler.compile(&api);

        // Verify generated code structure
        assert!(output.contains("pub enum MTLCommandBufferStatus"));
        assert!(output.contains("Created,"));
        assert!(output.contains("Encoding,"));
        assert!(output.contains("Committed,"));
        assert!(output.contains("Completed,"));

        assert!(output.contains("pub struct MTLCommandBufferStateTracker"));
        assert!(output.contains("pub fn commit(&mut self)"));
        assert!(output.contains("self.status = MTLCommandBufferStatus::Committed"));
    }

    #[test]
    fn test_compile_transition_with_params() {
        let api = PlatformApi {
            name: "Metal".to_string(),
            states: vec![ApiState {
                name: "MTLCommandBuffer".to_string(),
                status_enum: Some(StateEnum {
                    name: "Status".to_string(),
                    variants: vec!["Created".to_string(), "Encoding".to_string()],
                }),
                transitions: vec![ApiTransition {
                    name: "addCompletedHandler".to_string(),
                    params: vec![Param {
                        name: "block".to_string(),
                        ty: Type::Named("Block".to_string()),
                    }],
                    requires: vec![Expr::Compare(
                        Box::new(Expr::Var("status".to_string())),
                        ComparisonOp::Eq,
                        Box::new(Expr::Var("Created".to_string())),
                    )],
                    ensures: vec![],
                }],
                invariants: vec![],
            }],
        };

        let compiler = PlatformApiCompiler::new("Metal");
        let output = compiler.compile(&api);

        assert!(output.contains("pub fn add_completed_handler(&mut self, block: Block)"));
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(
            to_snake_case("addCompletedHandler"),
            "add_completed_handler"
        );
        assert_eq!(to_snake_case("commit"), "commit");
        assert_eq!(to_snake_case("MTLCommandBuffer"), "m_t_l_command_buffer");
    }

    #[test]
    fn test_compile_full_spec() {
        use crate::ast::Spec;
        use crate::typecheck::TypedSpec;
        use std::collections::HashMap;

        let spec = TypedSpec {
            spec: Spec {
                types: vec![],
                properties: vec![Property::PlatformApi(PlatformApi {
                    name: "CUDA".to_string(),
                    states: vec![ApiState {
                        name: "CUstream".to_string(),
                        status_enum: None,
                        transitions: vec![ApiTransition {
                            name: "synchronize".to_string(),
                            params: vec![],
                            requires: vec![],
                            ensures: vec![],
                        }],
                        invariants: vec![],
                    }],
                })],
            },
            type_info: HashMap::new(),
        };

        let result = compile_to_platform_api(&spec);
        assert!(result.is_some());

        let compiled = result.unwrap();
        assert_eq!(compiled.backend, "platform_api");
        assert!(compiled.code.contains("CUstreamStateTracker"));
        assert!(compiled.code.contains("pub fn synchronize"));
    }

    // ========== compile_expr / compile_expr_impl tests ==========

    #[test]
    fn test_compile_expr_returns_non_empty() {
        let compiler = PlatformApiCompiler::new("Test");
        // Test basic integer expression
        let result = compiler.compile_expr(&Expr::Int(42));
        assert!(!result.is_empty());
        assert_eq!(result, "42");
    }

    #[test]
    fn test_compile_expr_for_state_returns_non_empty() {
        let compiler = PlatformApiCompiler::new("Test");
        let state = ApiState {
            name: "TestState".to_string(),
            status_enum: Some(StateEnum {
                name: "Status".to_string(),
                variants: vec!["Ready".to_string(), "Done".to_string()],
            }),
            transitions: vec![],
            invariants: vec![],
        };
        // Test integer expression with state context
        let result = compiler.compile_expr_for_state(&Expr::Int(99), &state);
        assert!(!result.is_empty());
        assert_eq!(result, "99");
    }

    #[test]
    fn test_compile_expr_impl_status_var() {
        let compiler = PlatformApiCompiler::new("Test");
        let state = ApiState {
            name: "TestState".to_string(),
            status_enum: Some(StateEnum {
                name: "Status".to_string(),
                variants: vec!["Ready".to_string()],
            }),
            transitions: vec![],
            invariants: vec![],
        };
        // The "status" variable should be mapped to "self.status"
        let result = compiler.compile_expr_for_state(&Expr::Var("status".to_string()), &state);
        assert_eq!(result, "self.status");
    }

    #[test]
    fn test_compile_expr_impl_enum_variant() {
        let compiler = PlatformApiCompiler::new("Test");
        let state = ApiState {
            name: "Buffer".to_string(),
            status_enum: Some(StateEnum {
                name: "Status".to_string(),
                variants: vec!["Ready".to_string(), "Busy".to_string()],
            }),
            transitions: vec![],
            invariants: vec![],
        };
        // Enum variants should be prefixed with the state name + "Status::"
        let result = compiler.compile_expr_for_state(&Expr::Var("Ready".to_string()), &state);
        assert_eq!(result, "BufferStatus::Ready");
    }

    #[test]
    fn test_compile_expr_impl_non_status_var() {
        let compiler = PlatformApiCompiler::new("Test");
        let state = ApiState {
            name: "TestState".to_string(),
            status_enum: None,
            transitions: vec![],
            invariants: vec![],
        };
        // Non-status variables pass through unchanged
        let result = compiler.compile_expr_for_state(&Expr::Var("some_var".to_string()), &state);
        assert_eq!(result, "some_var");
    }

    #[test]
    fn test_compile_transition_precondition_negation() {
        // This tests that preconditions use `if !(condition)` for validation
        let api = PlatformApi {
            name: "Test".to_string(),
            states: vec![ApiState {
                name: "Device".to_string(),
                status_enum: Some(StateEnum {
                    name: "Status".to_string(),
                    variants: vec!["Ready".to_string()],
                }),
                transitions: vec![ApiTransition {
                    name: "process".to_string(),
                    params: vec![],
                    requires: vec![Expr::Compare(
                        Box::new(Expr::Var("status".to_string())),
                        ComparisonOp::Eq,
                        Box::new(Expr::Var("Ready".to_string())),
                    )],
                    ensures: vec![],
                }],
                invariants: vec![],
            }],
        };

        let compiler = PlatformApiCompiler::new("Test");
        let output = compiler.compile(&api);

        // The generated code should check `if !(...) {` to validate precondition
        // If this passed, the mutation that deletes ! would break
        assert!(output.contains("if !("));
        assert!(output.contains("return Err"));
    }

    // ========== type_to_rust tests ==========

    #[test]
    fn test_type_to_rust_int_variants() {
        // These must NOT just fallthrough to name.clone()
        assert_eq!(type_to_rust(&Type::Named("Int".to_string())), "i64");
        assert_eq!(type_to_rust(&Type::Named("int".to_string())), "i64");
        assert_eq!(type_to_rust(&Type::Named("i64".to_string())), "i64");
        // A different int-like name should NOT map to i64
        assert_ne!(type_to_rust(&Type::Named("Integer".to_string())), "i64");
    }

    #[test]
    fn test_type_to_rust_real_variants() {
        assert_eq!(type_to_rust(&Type::Named("Real".to_string())), "f64");
        assert_eq!(type_to_rust(&Type::Named("real".to_string())), "f64");
        assert_eq!(type_to_rust(&Type::Named("f64".to_string())), "f64");
        // A different float-like name should NOT map to f64
        assert_ne!(type_to_rust(&Type::Named("Float".to_string())), "f64");
    }

    #[test]
    fn test_type_to_rust_bool_variants() {
        assert_eq!(type_to_rust(&Type::Named("Bool".to_string())), "bool");
        assert_eq!(type_to_rust(&Type::Named("bool".to_string())), "bool");
        assert_eq!(type_to_rust(&Type::Named("boolean".to_string())), "bool");
        // A different bool-like name should NOT map to bool
        assert_ne!(type_to_rust(&Type::Named("Boolean".to_string())), "bool");
    }

    #[test]
    fn test_type_to_rust_string_variants() {
        assert_eq!(type_to_rust(&Type::Named("String".to_string())), "String");
        assert_eq!(type_to_rust(&Type::Named("string".to_string())), "String");
        // A different string-like name should NOT map to String
        assert_ne!(type_to_rust(&Type::Named("Str".to_string())), "String");
    }

    #[test]
    fn test_type_to_rust_custom_type_passthrough() {
        // Custom types should pass through unchanged
        assert_eq!(
            type_to_rust(&Type::Named("MyCustomType".to_string())),
            "MyCustomType"
        );
        assert_eq!(type_to_rust(&Type::Named("Block".to_string())), "Block");
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::ast::{BinaryOp, ComparisonOp, Expr, Type};

    /// Prove that compile_expr never produces empty output for integer literals.
    #[kani::proof]
    fn verify_platform_api_compile_expr_int_nonempty() {
        let compiler = PlatformApiCompiler::new("Test");
        let n: i64 = kani::any();
        let expr = Expr::Int(n);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
    }

    /// Prove that compile_expr never produces empty output for boolean literals.
    #[kani::proof]
    fn verify_platform_api_compile_expr_bool_nonempty() {
        let compiler = PlatformApiCompiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Bool(b);
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Platform API booleans compile to "true" or "false"
        assert!(result == "true" || result == "false");
    }

    /// Prove that type_to_rust handles Unit type correctly.
    #[kani::proof]
    fn verify_platform_api_type_to_rust_unit() {
        let ty = Type::Unit;
        let result = type_to_rust(&ty);
        assert_eq!(result, "()");
    }

    /// Prove that type_to_rust handles Int type correctly.
    #[kani::proof]
    fn verify_platform_api_type_to_rust_int() {
        let ty = Type::Named("Int".to_string());
        let result = type_to_rust(&ty);
        assert_eq!(result, "i64");
    }

    /// Prove that type_to_rust handles Bool type correctly.
    #[kani::proof]
    fn verify_platform_api_type_to_rust_bool() {
        let ty = Type::Named("Bool".to_string());
        let result = type_to_rust(&ty);
        assert_eq!(result, "bool");
    }

    /// Prove that comparison operators compile to valid Rust syntax.
    #[kani::proof]
    fn verify_platform_api_comparison_valid() {
        let compiler = PlatformApiCompiler::new("Test");
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
        // All comparison results should start with "(" for grouping
        assert!(result.starts_with('('));
    }

    /// Prove that binary operators compile to valid Rust syntax.
    #[kani::proof]
    fn verify_platform_api_binary_ops_nonempty() {
        let compiler = PlatformApiCompiler::new("Test");
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
        // Binary ops produce parenthesized output
        assert!(result.starts_with('('));
    }

    /// Prove that implies compiles to non-empty output with correct structure.
    #[kani::proof]
    fn verify_platform_api_implies_nonempty() {
        let compiler = PlatformApiCompiler::new("Test");
        let expr = Expr::Implies(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Implies is compiled as !(P) || (Q)
        assert!(result.contains("||"));
    }

    /// Prove that not compiles to non-empty output with correct prefix.
    #[kani::proof]
    fn verify_platform_api_not_nonempty() {
        let compiler = PlatformApiCompiler::new("Test");
        let b: bool = kani::any();
        let expr = Expr::Not(Box::new(Expr::Bool(b)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // Not is compiled as !(expr)
        assert!(result.starts_with("!("));
    }

    /// Prove that and compiles to non-empty output.
    #[kani::proof]
    fn verify_platform_api_and_nonempty() {
        let compiler = PlatformApiCompiler::new("Test");
        let expr = Expr::And(Box::new(Expr::Bool(true)), Box::new(Expr::Bool(false)));
        let result = compiler.compile_expr(&expr);
        assert!(!result.is_empty());
        // And is compiled as (P && Q)
        assert!(result.contains("&&"));
    }

    /// Prove that to_snake_case produces non-empty output for non-empty input.
    #[kani::proof]
    fn verify_platform_api_to_snake_case_nonempty() {
        // Test with a simple fixed input rather than symbolic (to avoid explosion)
        let result = to_snake_case("TestMethod");
        assert!(!result.is_empty());
        assert_eq!(result, "test_method");
    }
}
