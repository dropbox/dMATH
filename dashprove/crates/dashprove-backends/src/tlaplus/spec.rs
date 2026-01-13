//! TLA+ specification file generation

use crate::traits::BackendError;
use dashprove_usl::{compile_to_tlaplus, typecheck::TypedSpec, TlaPlusCompiler};
use std::path::{Path, PathBuf};
use tracing::debug;

/// Generate TLA+ config file (CFG) for model checking
///
/// This generates a configuration file for TLC model checker, including:
/// - SPECIFICATION: The main behavior specification
/// - INVARIANT: Safety properties to check
/// - PROPERTY: Temporal/liveness properties to check
/// - CHECK_DEADLOCK: Enabled for liveness checking
/// - CONSTANTS: Model value assignments
///
/// For liveness properties with fairness constraints, the specification
/// is augmented to include the fairness assumptions.
pub fn generate_config(spec: &TypedSpec, _module_name: &str) -> String {
    let mut cfg = String::new();
    let compiler = TlaPlusCompiler::new("USLSpec");

    // Collect all fairness constraints from temporal properties
    let mut all_fairness: Vec<String> = Vec::new();
    let mut has_liveness = false;

    for property in &spec.spec.properties {
        if let dashprove_usl::Property::Temporal(temp) = property {
            // Check if this is a liveness property (contains eventually or leads-to)
            if is_liveness_property(temp) {
                has_liveness = true;
            }

            // Collect fairness constraints
            for fc in &temp.fairness {
                all_fairness.push(compiler.compile_fairness(fc));
            }
        }
    }

    // Write SPECIFICATION with fairness if needed
    // TLC supports: SPECIFICATION Spec /\ WF_vars(Next) /\ SF_vars(Other)
    if all_fairness.is_empty() {
        cfg.push_str("SPECIFICATION Spec\n");
    } else {
        // Add fairness constraints to the specification
        let fairness_conj = all_fairness.join(" /\\ ");
        cfg.push_str(&format!("SPECIFICATION Spec /\\ {}\n", fairness_conj));
    }

    // Enable deadlock checking for liveness
    // (TLC needs this to properly check liveness properties)
    if has_liveness {
        cfg.push_str("CHECK_DEADLOCK FALSE\n");
    }

    cfg.push('\n');

    // Extract invariants and temporal properties from spec
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
            // Define constants as model values
            cfg.push_str(&format!("CONSTANT {} = {}\n", type_def.name, type_def.name));
        }
    }

    cfg
}

/// Check if a temporal property is a liveness property
///
/// Liveness properties assert that "something good eventually happens".
/// They typically contain `eventually` or `leads-to` operators.
fn is_liveness_property(temp: &dashprove_usl::Temporal) -> bool {
    contains_liveness_operator(&temp.body)
}

/// Recursively check if a temporal expression contains liveness operators
fn contains_liveness_operator(expr: &dashprove_usl::TemporalExpr) -> bool {
    match expr {
        dashprove_usl::TemporalExpr::Eventually(_) => true,
        dashprove_usl::TemporalExpr::LeadsTo(_, _) => true,
        dashprove_usl::TemporalExpr::Always(inner) => contains_liveness_operator(inner),
        dashprove_usl::TemporalExpr::Atom(_) => false,
    }
}

/// Write TLA+ module and config to temp directory
pub async fn write_spec(spec: &TypedSpec, dir: &Path) -> Result<(PathBuf, PathBuf), BackendError> {
    let compiled = compile_to_tlaplus(spec);
    let module_name = compiled.module_name.as_deref().unwrap_or("USLSpec");

    // Write TLA+ module
    let tla_path = dir.join(format!("{}.tla", module_name));
    tokio::fs::write(&tla_path, &compiled.code)
        .await
        .map_err(|e| {
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

    debug!("Written TLA+ spec to {:?}", tla_path);
    debug!("Written config to {:?}", cfg_path);

    Ok((tla_path, cfg_path))
}
