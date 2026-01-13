//! Code generation for runtime monitors in different target languages

use crate::monitor::compile_expr::{
    compile_expr_to_python, compile_expr_to_python_with_env, compile_expr_to_rust,
    compile_expr_to_rust_with_env, compile_expr_to_typescript, compile_expr_to_typescript_with_env,
    compile_temporal_to_rust,
};
use crate::monitor::config::{MonitorConfig, MonitorTarget};
use crate::monitor::params::{
    join_params, python_contract_params, rust_contract_params, ts_contract_params,
};
use crate::monitor::property::{contract_base_name, property_expr, property_temporal_expr};
use crate::monitor::utils::to_pascal_case;
use dashprove_usl::ast::{Expr, Property};
use std::collections::HashMap;

/// Generate complete monitor code with real expression compilation
pub fn generate_monitor_code_with_exprs(
    name: &str,
    properties: &[Property],
    config: &MonitorConfig,
) -> String {
    match config.target {
        MonitorTarget::Rust => generate_rust_monitor_with_exprs(name, properties, config),
        MonitorTarget::TypeScript => {
            generate_typescript_monitor_with_exprs(name, properties, config)
        }
        MonitorTarget::Python => generate_python_monitor_with_exprs(name, properties, config),
    }
}

/// Generate check code for a single property
pub fn generate_check_code(property_name: &str, config: &MonitorConfig) -> String {
    match config.target {
        MonitorTarget::Rust => {
            if config.generate_assertions {
                format!(
                    r#"assert!(self.check_{name}(), "Property '{name}' violated");"#,
                    name = property_name
                )
            } else if config.generate_logging {
                format!(
                    r#"if !self.check_{name}() {{ tracing::error!("Property '{name}' violated"); }}"#,
                    name = property_name
                )
            } else {
                format!("self.check_{name}();", name = property_name)
            }
        }
        MonitorTarget::TypeScript => {
            format!(
                r#"if (!this.check{name}()) {{ console.error("Property '{name}' violated"); }}"#,
                name = to_pascal_case(property_name)
            )
        }
        MonitorTarget::Python => {
            format!(
                r#"if not self.check_{name}():
    logging.error("Property '{name}' violated")"#,
                name = property_name
            )
        }
    }
}

// =============================================================================
// Rust Code Generation
// =============================================================================

fn generate_rust_monitor_with_exprs(
    name: &str,
    properties: &[Property],
    config: &MonitorConfig,
) -> String {
    let mut check_methods = Vec::new();
    let mut zero_arg_checks = Vec::new();

    for property in properties {
        match property {
            Property::Contract(contract) => {
                let base = contract_base_name(contract);
                let prop_name = contract.type_path.join("::");
                let (params, result_param, env) = rust_contract_params(contract);

                if !contract.requires.is_empty() {
                    let requires_body = join_rust_expressions(&contract.requires, &env);
                    let param_list = join_params(&params, None);
                    check_methods.push(format!(
                        r#"
    /// Check preconditions for {prop_name}
    pub fn check_{base}_requires(&self{comma}{params}) -> bool {{
        {requires_body}
    }}"#,
                        prop_name = prop_name,
                        base = base,
                        comma = if param_list.is_empty() { "" } else { ", " },
                        params = param_list,
                        requires_body = requires_body
                    ));
                }

                if !contract.ensures.is_empty() {
                    let ensures_body = join_rust_expressions(&contract.ensures, &env);
                    let params_with_result = join_params(&params, result_param.as_ref());
                    check_methods.push(format!(
                        r#"
    /// Check success postconditions for {prop_name}
    pub fn check_{base}_ensures(&self{comma}{params}) -> bool {{
        {ensures_body}
    }}"#,
                        prop_name = prop_name,
                        base = base,
                        comma = if params_with_result.is_empty() {
                            ""
                        } else {
                            ", "
                        },
                        params = params_with_result,
                        ensures_body = ensures_body
                    ));
                }

                if !contract.ensures_err.is_empty() {
                    let ensures_err_body = join_rust_expressions(&contract.ensures_err, &env);
                    let params_with_result = join_params(&params, result_param.as_ref());
                    check_methods.push(format!(
                        r#"
    /// Check error postconditions for {prop_name}
    pub fn check_{base}_ensures_err(&self{comma}{params}) -> bool {{
        {ensures_err_body}
    }}"#,
                        prop_name = prop_name,
                        base = base,
                        comma = if params_with_result.is_empty() {
                            ""
                        } else {
                            ", "
                        },
                        params = params_with_result,
                        ensures_err_body = ensures_err_body
                    ));
                }
            }
            _ => {
                let prop_name = property.name();
                let body_code = if let Some(expr) = property_expr(property) {
                    compile_expr_to_rust(expr)
                } else if let Some(temporal) = property_temporal_expr(property) {
                    compile_temporal_to_rust(temporal)
                } else {
                    "true".to_string()
                };
                zero_arg_checks.push(format!("        self.check_{name}()", name = prop_name));
                check_methods.push(format!(
                    r#"
    /// Check property: {name}
    pub fn check_{name}(&self) -> bool {{
        {body}
    }}"#,
                    name = prop_name,
                    body = body_code
                ));
            }
        }
    }

    let check_methods = check_methods.join("\n");

    let check_all = if zero_arg_checks.is_empty() {
        "        true".to_string()
    } else {
        zero_arg_checks.join(" &&\n")
    };

    let mut code = format!(
        r#"//! Runtime monitor for {name}
//! Generated by DashProve

/// Runtime monitor for specification: {name}
pub struct {struct_name}Monitor {{
    // State fields would go here
}}

impl {struct_name}Monitor {{
    /// Create a new monitor
    pub fn new() -> Self {{
        Self {{}}
    }}
{check_methods}

    /// Check all properties
    pub fn check_all(&self) -> bool {{
{check_all}
    }}
}}

impl Default for {struct_name}Monitor {{
    fn default() -> Self {{
        Self::new()
    }}
}}
"#,
        name = name,
        struct_name = to_pascal_case(name),
        check_methods = check_methods,
        check_all = check_all,
    );

    if config.generate_logging {
        code = format!("use tracing;\n\n{}", code);
    }

    code
}

// =============================================================================
// TypeScript Code Generation
// =============================================================================

fn generate_typescript_monitor_with_exprs(
    name: &str,
    properties: &[Property],
    _config: &MonitorConfig,
) -> String {
    let mut check_methods = Vec::new();
    let mut zero_arg_checks = Vec::new();

    for property in properties {
        match property {
            Property::Contract(contract) => {
                let base = contract_base_name(contract);
                let pascal = to_pascal_case(&base);
                let (params, result_param, env) = ts_contract_params(contract);
                let prop_name = contract.type_path.join("::");

                if !contract.requires.is_empty() {
                    let requires_body = join_ts_expressions(&contract.requires, &env);
                    let param_list = join_params(&params, None);
                    check_methods.push(format!(
                        r#"
  /** Check preconditions for {prop_name} */
  check{pascal}Requires({params}): boolean {{
    return {requires_body};
  }}"#,
                        prop_name = prop_name,
                        pascal = pascal,
                        params = param_list,
                        requires_body = requires_body
                    ));
                }

                if !contract.ensures.is_empty() {
                    let ensures_body = join_ts_expressions(&contract.ensures, &env);
                    let params_with_result = join_params(&params, result_param.as_ref());
                    check_methods.push(format!(
                        r#"
  /** Check success postconditions for {prop_name} */
  check{pascal}Ensures({params}): boolean {{
    return {ensures_body};
  }}"#,
                        prop_name = prop_name,
                        pascal = pascal,
                        params = params_with_result,
                        ensures_body = ensures_body
                    ));
                }

                if !contract.ensures_err.is_empty() {
                    let ensures_err_body = join_ts_expressions(&contract.ensures_err, &env);
                    let params_with_result = join_params(&params, result_param.as_ref());
                    check_methods.push(format!(
                        r#"
  /** Check error postconditions for {prop_name} */
  check{pascal}EnsuresErr({params}): boolean {{
    return {ensures_err_body};
  }}"#,
                        prop_name = prop_name,
                        pascal = pascal,
                        params = params_with_result,
                        ensures_err_body = ensures_err_body
                    ));
                }
            }
            _ => {
                let prop_name = property.name();
                let body_code = if let Some(expr) = property_expr(property) {
                    compile_expr_to_typescript(expr)
                } else {
                    "true".to_string()
                };
                zero_arg_checks.push(format!("      this.check{}()", to_pascal_case(&prop_name)));
                check_methods.push(format!(
                    r#"
  /** Check property: {name} */
  check{pascal}(): boolean {{
    return {body};
  }}"#,
                    name = prop_name,
                    pascal = to_pascal_case(&prop_name),
                    body = body_code
                ));
            }
        }
    }

    let check_methods = check_methods.join("\n");

    let check_all = if zero_arg_checks.is_empty() {
        "      true".to_string()
    } else {
        zero_arg_checks.join(" &&\n")
    };

    format!(
        r#"/**
 * Runtime monitor for {name}
 * Generated by DashProve
 */
export class {struct_name}Monitor {{
  constructor() {{
    // Initialize state
  }}
{check_methods}

  /** Check all properties */
  checkAll(): boolean {{
    return (
{check_all}
    );
  }}
}}
"#,
        name = name,
        struct_name = to_pascal_case(name),
        check_methods = check_methods,
        check_all = check_all,
    )
}

// =============================================================================
// Python Code Generation
// =============================================================================

fn generate_python_monitor_with_exprs(
    name: &str,
    properties: &[Property],
    _config: &MonitorConfig,
) -> String {
    let mut check_methods = Vec::new();
    let mut zero_arg_checks = Vec::new();

    for property in properties {
        match property {
            Property::Contract(contract) => {
                let base = contract_base_name(contract);
                let (params, result_param, env) = python_contract_params(contract);
                let prop_name = contract.type_path.join("::");

                if !contract.requires.is_empty() {
                    let requires_body = join_python_expressions(&contract.requires, &env);
                    let param_list = join_params(&params, None);
                    let params_with_self = if param_list.is_empty() {
                        "self".to_string()
                    } else {
                        format!("self, {param_list}")
                    };
                    check_methods.push(format!(
                        r#"
    def check_{name}_requires({params}) -> bool:
        """Check preconditions for {prop_name}"""
        return {requires_body}"#,
                        name = base,
                        params = params_with_self,
                        prop_name = prop_name,
                        requires_body = requires_body
                    ));
                }

                if !contract.ensures.is_empty() {
                    let ensures_body = join_python_expressions(&contract.ensures, &env);
                    let params_with_result = join_params(&params, result_param.as_ref());
                    let params_with_self = if params_with_result.is_empty() {
                        "self".to_string()
                    } else {
                        format!("self, {params_with_result}")
                    };
                    check_methods.push(format!(
                        r#"
    def check_{name}_ensures({params}) -> bool:
        """Check success postconditions for {prop_name}"""
        return {ensures_body}"#,
                        name = base,
                        params = params_with_self,
                        prop_name = prop_name,
                        ensures_body = ensures_body
                    ));
                }

                if !contract.ensures_err.is_empty() {
                    let ensures_err_body = join_python_expressions(&contract.ensures_err, &env);
                    let params_with_result = join_params(&params, result_param.as_ref());
                    let params_with_self = if params_with_result.is_empty() {
                        "self".to_string()
                    } else {
                        format!("self, {params_with_result}")
                    };
                    check_methods.push(format!(
                        r#"
    def check_{name}_ensures_err({params}) -> bool:
        """Check error postconditions for {prop_name}"""
        return {ensures_err_body}"#,
                        name = base,
                        params = params_with_self,
                        prop_name = prop_name,
                        ensures_err_body = ensures_err_body
                    ));
                }
            }
            _ => {
                let prop_name = property.name();
                let body_code = if let Some(expr) = property_expr(property) {
                    compile_expr_to_python(expr)
                } else {
                    "True".to_string()
                };
                zero_arg_checks.push(format!("            self.check_{name}()", name = prop_name));
                check_methods.push(format!(
                    r#"
    def check_{name}(self) -> bool:
        """Check property: {name}"""
        return {body}"#,
                    name = prop_name,
                    body = body_code
                ));
            }
        }
    }

    let check_methods = check_methods.join("\n");

    let check_all = if zero_arg_checks.is_empty() {
        "            True".to_string()
    } else {
        zero_arg_checks.join(" and\n")
    };

    format!(
        r#"\"\"\"
Runtime monitor for {name}
Generated by DashProve
\"\"\"
import logging


class {struct_name}Monitor:
    \"\"\"Runtime monitor for specification: {name}\"\"\"

    def __init__(self):
        \"\"\"Create a new monitor\"\"\"
        pass
{check_methods}

    def check_all(self) -> bool:
        \"\"\"Check all properties\"\"\"
        return (
{check_all}
        )
"#,
        name = name,
        struct_name = to_pascal_case(name),
        check_methods = check_methods,
        check_all = check_all,
    )
}

// =============================================================================
// Expression Joining Helpers
// =============================================================================

fn join_rust_expressions(exprs: &[Expr], env: &HashMap<String, String>) -> String {
    let compiled: Vec<String> = exprs
        .iter()
        .map(|expr| compile_expr_to_rust_with_env(expr, env))
        .collect();

    match compiled.len() {
        0 => "true".to_string(),
        1 => compiled[0].clone(),
        _ => compiled.join(" &&\n        "),
    }
}

fn join_ts_expressions(exprs: &[Expr], env: &HashMap<String, String>) -> String {
    let compiled: Vec<String> = exprs
        .iter()
        .map(|expr| compile_expr_to_typescript_with_env(expr, env))
        .collect();

    match compiled.len() {
        0 => "true".to_string(),
        1 => compiled[0].clone(),
        _ => compiled.join(" &&\n      "),
    }
}

fn join_python_expressions(exprs: &[Expr], env: &HashMap<String, String>) -> String {
    let compiled: Vec<String> = exprs
        .iter()
        .map(|expr| compile_expr_to_python_with_env(expr, env))
        .collect();

    match compiled.len() {
        0 => "True".to_string(),
        1 => compiled[0].clone(),
        _ => compiled.join("\n        and "),
    }
}
