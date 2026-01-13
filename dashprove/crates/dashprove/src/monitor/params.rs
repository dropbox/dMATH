//! Parameter handling for contract code generation

use crate::monitor::types::{rust_type_name, ts_type_name};
use crate::monitor::utils::{sanitize_identifier, to_camel_case, to_snake_case};
use dashprove_usl::ast::Contract;
use std::collections::HashMap;

/// Generate Rust parameter name from USL identifier
pub fn rust_param_name(name: &str) -> String {
    let base = to_snake_case(&sanitize_identifier(name));
    if base.is_empty() {
        "value".to_string()
    } else if base == "self" {
        "self_value".to_string()
    } else {
        base
    }
}

/// Generate TypeScript parameter name from USL identifier
pub fn ts_param_name(name: &str) -> String {
    let base = to_camel_case(&to_snake_case(&sanitize_identifier(name)));
    if base.is_empty() {
        "value".to_string()
    } else if base == "self" {
        "selfValue".to_string()
    } else {
        base
    }
}

/// Generate Python parameter name from USL identifier
pub fn python_param_name(name: &str) -> String {
    let base = to_snake_case(&sanitize_identifier(name));
    if base.is_empty() {
        "value".to_string()
    } else if base == "self" {
        "self_value".to_string()
    } else {
        base
    }
}

/// Extract Rust contract parameters
/// Returns (params, result_param, env) where:
/// - params: list of "name: Type" strings
/// - result_param: optional result parameter string
/// - env: mapping from USL names to Rust names
pub fn rust_contract_params(
    contract: &Contract,
) -> (Vec<String>, Option<String>, HashMap<String, String>) {
    let mut env = HashMap::new();
    let mut params = Vec::new();

    for param in &contract.params {
        let name = rust_param_name(&param.name);
        env.insert(param.name.clone(), name.clone());
        params.push(format!("{name}: {}", rust_type_name(&param.ty)));
    }

    let result_param = contract.return_type.as_ref().map(|ty| {
        let name = rust_param_name("result");
        env.insert("result".to_string(), name.clone());
        format!("{name}: {}", rust_type_name(ty))
    });

    (params, result_param, env)
}

/// Extract TypeScript contract parameters
pub fn ts_contract_params(
    contract: &Contract,
) -> (Vec<String>, Option<String>, HashMap<String, String>) {
    let mut env = HashMap::new();
    let mut params = Vec::new();

    for param in &contract.params {
        let name = ts_param_name(&param.name);
        env.insert(param.name.clone(), name.clone());
        params.push(format!("{name}: {}", ts_type_name(&param.ty)));
    }

    let result_param = contract.return_type.as_ref().map(|ty| {
        let name = ts_param_name("result");
        env.insert("result".to_string(), name.clone());
        format!("{name}: {}", ts_type_name(ty))
    });

    (params, result_param, env)
}

/// Extract Python contract parameters
pub fn python_contract_params(
    contract: &Contract,
) -> (Vec<String>, Option<String>, HashMap<String, String>) {
    let mut env = HashMap::new();
    let mut params = Vec::new();

    for param in &contract.params {
        let name = python_param_name(&param.name);
        env.insert(param.name.clone(), name.clone());
        params.push(name);
    }

    let result_param = contract.return_type.as_ref().map(|_| {
        let name = python_param_name("result");
        env.insert("result".to_string(), name.clone());
        name
    });

    (params, result_param, env)
}

/// Join parameters into a comma-separated string, optionally including result
pub fn join_params(params: &[String], result_param: Option<&String>) -> String {
    let mut combined = params.to_owned();
    if let Some(result) = result_param {
        combined.push(result.clone());
    }
    combined.join(", ")
}
