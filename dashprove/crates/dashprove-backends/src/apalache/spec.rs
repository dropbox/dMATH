//! Apalache specification file generation with type annotations
//!
//! Apalache requires type annotations for symbolic model checking.
//! This module generates TLA+ specs with Apalache type annotations.

use crate::traits::BackendError;
use dashprove_usl::typecheck::TypedSpec;
use dashprove_usl::{compile_to_tlaplus, TlaPlusCompiler, Type};
use std::path::{Path, PathBuf};
use tracing::debug;

/// Type annotation generator for Apalache
pub struct TypeAnnotator;

impl TypeAnnotator {
    /// Generate Apalache type annotations for a TLA+ spec
    ///
    /// Apalache uses annotations in the form:
    /// - `\* @type: Int;` for variable types
    /// - `\* @typeAlias: STATE = [x: Int, y: Set(Int)];` for type aliases
    ///
    /// We infer types from the USL specification.
    pub fn annotate(spec: &TypedSpec, tlaplus_code: &str) -> String {
        let mut result = String::new();

        // Add type aliases at the module header
        result.push_str("---- MODULE USLSpec_Apalache ----\n");
        result.push_str("\\* Apalache type annotations\n\n");

        // Generate type aliases from USL type definitions
        for type_def in &spec.spec.types {
            // Convert TypeDef fields to record type annotation
            let field_types: Vec<String> = type_def
                .fields
                .iter()
                .map(|f| format!("{}: {}", f.name, Self::usl_type_to_apalache(&f.ty)))
                .collect();
            let alias = if field_types.is_empty() {
                "Int".to_string() // Default for empty types
            } else {
                format!("[{}]", field_types.join(", "))
            };
            result.push_str(&format!("\\* @typeAlias: {} = {};\n", type_def.name, alias));
        }
        result.push('\n');

        // Skip the module header from the original TLA+ code
        // and add the rest with type annotations
        let mut in_variables = false;
        let mut in_constants = false;

        for line in tlaplus_code.lines() {
            let trimmed = line.trim();

            // Skip original module header
            if trimmed.starts_with("---- MODULE") || trimmed.starts_with("====") {
                if trimmed.starts_with("====") {
                    result.push_str(line);
                    result.push('\n');
                }
                continue;
            }

            // Track VARIABLES section
            if trimmed.starts_with("VARIABLES") {
                in_variables = true;
                in_constants = false;
            } else if trimmed.starts_with("CONSTANTS") {
                in_constants = true;
                in_variables = false;
            } else if trimmed.is_empty() || trimmed.starts_with("(*") {
                in_variables = false;
                in_constants = false;
            }

            // Add type annotations for variable declarations
            if in_variables && !trimmed.starts_with("VARIABLES") && !trimmed.is_empty() {
                // Parse variable name and add type annotation
                let vars: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();
                for var in vars {
                    if !var.is_empty() {
                        // Try to infer type from spec
                        let var_type = Self::infer_variable_type(spec, var);
                        result.push_str(&format!("\\* @type: {};\n", var_type));
                    }
                }
            }

            // Add type annotations for constants
            if in_constants && !trimmed.starts_with("CONSTANTS") && !trimmed.is_empty() {
                let consts: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();
                for constant in consts {
                    if !constant.is_empty() {
                        let const_type = Self::infer_constant_type(spec, constant);
                        result.push_str(&format!("\\* @type: {};\n", const_type));
                    }
                }
            }

            result.push_str(line);
            result.push('\n');
        }

        // Add module footer if not present
        if !result.ends_with("====\n") {
            result.push_str("====\n");
        }

        result
    }

    /// Convert USL type to Apalache type annotation string
    fn usl_type_to_apalache(usl_type: &Type) -> String {
        match usl_type {
            Type::Named(name) => {
                // Map common types
                match name.as_str() {
                    "Int" | "int" | "integer" => "Int".to_string(),
                    "Nat" | "nat" | "natural" => "Int".to_string(), // Apalache uses Int
                    "Bool" | "bool" | "boolean" => "Bool".to_string(),
                    "String" | "string" => "Str".to_string(),
                    "Real" | "real" => "Int".to_string(), // Apalache doesn't support reals
                    _ => name.clone(),
                }
            }
            Type::Set(inner) => {
                format!("Set({})", Self::usl_type_to_apalache(inner))
            }
            Type::List(inner) => {
                format!("Seq({})", Self::usl_type_to_apalache(inner))
            }
            Type::Map(key, value) => {
                format!(
                    "{} -> {}",
                    Self::usl_type_to_apalache(key),
                    Self::usl_type_to_apalache(value)
                )
            }
            Type::Relation(left, right) => {
                // Relations are sets of tuples in Apalache
                format!(
                    "Set(<<{}, {}>>)",
                    Self::usl_type_to_apalache(left),
                    Self::usl_type_to_apalache(right)
                )
            }
            Type::Function(arg, ret) => {
                format!(
                    "{} => {}",
                    Self::usl_type_to_apalache(arg),
                    Self::usl_type_to_apalache(ret)
                )
            }
            Type::Result(inner) => {
                // Result as variant in Apalache
                format!("Ok({}) | Err(Str)", Self::usl_type_to_apalache(inner))
            }
            Type::Unit => "Bool".to_string(), // Unit represented as Bool in TLA+
            Type::Graph(n, e) => {
                format!(
                    "[nodes: Set({}), edges: Set(<<{}, {}>>)]",
                    Self::usl_type_to_apalache(n),
                    Self::usl_type_to_apalache(n),
                    Self::usl_type_to_apalache(e)
                )
            }
            Type::Path(n) => format!("Seq({})", Self::usl_type_to_apalache(n)),
        }
    }

    /// Infer type for a variable from USL spec
    fn infer_variable_type(spec: &TypedSpec, var_name: &str) -> String {
        // Look for variable in type definitions (which act as state shape)
        for type_def in &spec.spec.types {
            for field in &type_def.fields {
                if field.name == var_name {
                    return Self::usl_type_to_apalache(&field.ty);
                }
            }
        }

        // Default to Int for unknown types
        "Int".to_string()
    }

    /// Infer type for a constant from USL spec
    fn infer_constant_type(spec: &TypedSpec, const_name: &str) -> String {
        // Look for constant in type definitions
        for type_def in &spec.spec.types {
            if type_def.name == const_name {
                // Return a record type for the entire type definition
                let field_types: Vec<String> = type_def
                    .fields
                    .iter()
                    .map(|f| format!("{}: {}", f.name, Self::usl_type_to_apalache(&f.ty)))
                    .collect();
                if field_types.is_empty() {
                    return "Int".to_string();
                }
                return format!("[{}]", field_types.join(", "));
            }
        }

        // Default to Int for unknown
        "Int".to_string()
    }
}

/// Generate Apalache config file
///
/// Unlike TLC config, Apalache uses command-line args for most settings,
/// but supports a TLC-compatible config file for specification and properties.
pub fn generate_config(spec: &TypedSpec, _module_name: &str) -> String {
    let mut cfg = String::new();
    let compiler = TlaPlusCompiler::new("USLSpec_Apalache");

    // Collect all fairness constraints from temporal properties
    let mut all_fairness: Vec<String> = Vec::new();
    let mut has_liveness = false;

    for property in &spec.spec.properties {
        if let dashprove_usl::Property::Temporal(temp) = property {
            // Check if this is a liveness property
            if contains_liveness_operator(&temp.body) {
                has_liveness = true;
            }

            // Collect fairness constraints
            for fc in &temp.fairness {
                all_fairness.push(compiler.compile_fairness(fc));
            }
        }
    }

    // Write SPECIFICATION with fairness if needed
    if all_fairness.is_empty() {
        cfg.push_str("SPECIFICATION Spec\n");
    } else {
        let fairness_conj = all_fairness.join(" /\\ ");
        cfg.push_str(&format!("SPECIFICATION Spec /\\ {}\n", fairness_conj));
    }

    // Enable deadlock checking for liveness
    if has_liveness {
        cfg.push_str("CHECK_DEADLOCK FALSE\n");
    }

    cfg.push('\n');

    // Extract invariants and temporal properties
    for property in &spec.spec.properties {
        match property {
            dashprove_usl::Property::Invariant(inv) => {
                cfg.push_str(&format!("INVARIANT {}\n", inv.name));
            }
            dashprove_usl::Property::Temporal(temp) => {
                cfg.push_str(&format!("PROPERTY {}\n", temp.name));
            }
            _ => {}
        }
    }

    // Add constants if any type definitions exist
    if !spec.spec.types.is_empty() {
        cfg.push('\n');
        for type_def in &spec.spec.types {
            cfg.push_str(&format!("CONSTANT {} = {}\n", type_def.name, type_def.name));
        }
    }

    cfg
}

/// Check if a temporal expression contains liveness operators
fn contains_liveness_operator(expr: &dashprove_usl::TemporalExpr) -> bool {
    match expr {
        dashprove_usl::TemporalExpr::Eventually(_) => true,
        dashprove_usl::TemporalExpr::LeadsTo(_, _) => true,
        dashprove_usl::TemporalExpr::Always(inner) => contains_liveness_operator(inner),
        dashprove_usl::TemporalExpr::Atom(_) => false,
    }
}

/// Write Apalache TLA+ module (with type annotations) and config to temp directory
pub async fn write_spec(spec: &TypedSpec, dir: &Path) -> Result<(PathBuf, PathBuf), BackendError> {
    // First compile to standard TLA+
    let compiled = compile_to_tlaplus(spec);
    let module_name = "USLSpec_Apalache";

    // Add Apalache type annotations
    let annotated = TypeAnnotator::annotate(spec, &compiled.code);

    // Write TLA+ module with type annotations
    let tla_path = dir.join(format!("{}.tla", module_name));
    tokio::fs::write(&tla_path, &annotated).await.map_err(|e| {
        BackendError::CompilationFailed(format!("Failed to write TLA+ file: {}", e))
    })?;

    // Write config file
    let cfg_content = generate_config(spec, module_name);
    let cfg_path = dir.join(format!("{}.cfg", module_name));
    tokio::fs::write(&cfg_path, &cfg_content)
        .await
        .map_err(|e| {
            BackendError::CompilationFailed(format!("Failed to write config file: {}", e))
        })?;

    debug!("Written Apalache TLA+ spec to {:?}", tla_path);
    debug!("Written config to {:?}", cfg_path);

    Ok((tla_path, cfg_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usl_type_to_apalache_int() {
        let ty = Type::Named("Int".to_string());
        assert_eq!(TypeAnnotator::usl_type_to_apalache(&ty), "Int");
    }

    #[test]
    fn test_usl_type_to_apalache_set() {
        let ty = Type::Set(Box::new(Type::Named("Int".to_string())));
        assert_eq!(TypeAnnotator::usl_type_to_apalache(&ty), "Set(Int)");
    }

    #[test]
    fn test_usl_type_to_apalache_map() {
        let ty = Type::Map(
            Box::new(Type::Named("String".to_string())),
            Box::new(Type::Named("Int".to_string())),
        );
        assert_eq!(TypeAnnotator::usl_type_to_apalache(&ty), "Str -> Int");
    }

    #[test]
    fn test_usl_type_to_apalache_list() {
        let ty = Type::List(Box::new(Type::Named("Bool".to_string())));
        assert_eq!(TypeAnnotator::usl_type_to_apalache(&ty), "Seq(Bool)");
    }

    #[test]
    fn test_usl_type_to_apalache_function() {
        let ty = Type::Function(
            Box::new(Type::Named("Int".to_string())),
            Box::new(Type::Named("Bool".to_string())),
        );
        assert_eq!(TypeAnnotator::usl_type_to_apalache(&ty), "Int => Bool");
    }
}
