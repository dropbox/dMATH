//! Zonotope specification generation for ERAN
//!
//! Generates zonotope CSV specifications from USL specs for complex input regions.

use super::config::EranDomain;
use super::usl_analysis::{
    consolidate_bounds, extract_epsilon, extract_input_bounds, extract_neural_dimensions,
};
use dashprove_usl::ast::Property;
use dashprove_usl::typecheck::TypedSpec;
use std::collections::HashSet;

/// Generate zonotope specification from USL spec
///
/// ERAN accepts zonotope specifications for complex input regions.
/// Format: CSV with headers, then rows of dimension, center, radius
pub fn generate_zonotope_spec(spec: &TypedSpec) -> String {
    let mut inputs = HashSet::new();
    let mut outputs = HashSet::new();
    let mut all_bounds = Vec::new();

    // Extract dimensions and bounds from type definitions
    for typedef in &spec.spec.types {
        for field in &typedef.fields {
            let lower = field.name.to_lowercase();
            if lower.starts_with("x") || lower.contains("input") {
                if let Some(idx) = super::usl_analysis::extract_index(&field.name) {
                    inputs.insert(idx);
                }
            } else if lower.starts_with("y") || lower.contains("output") {
                if let Some(idx) = super::usl_analysis::extract_index(&field.name) {
                    outputs.insert(idx);
                }
            }
        }
    }

    // Extract from properties
    for prop in &spec.spec.properties {
        let expr = match prop {
            Property::Invariant(inv) => Some(&inv.body),
            Property::Theorem(thm) => Some(&thm.body),
            Property::Security(sec) => Some(&sec.body),
            Property::Probabilistic(prob) => Some(&prob.condition),
            _ => None,
        };
        if let Some(e) = expr {
            extract_neural_dimensions(e, &mut inputs, &mut outputs);
            let prop_bounds = extract_input_bounds(e);
            all_bounds.extend(prop_bounds);
        }
    }

    let consolidated = consolidate_bounds(&all_bounds);

    // Generate zonotope spec
    // Format: dim,center,epsilon
    let mut spec_lines = Vec::new();
    spec_lines.push("dim,center,epsilon".to_string());

    let max_input = inputs.iter().max().copied().unwrap_or(0);
    for i in 0..=max_input {
        // Find bounds for this input, or use defaults
        let (lower, upper) = consolidated
            .iter()
            .find(|(idx, _, _)| *idx == i)
            .map(|(_, l, u)| (*l, *u))
            .unwrap_or((0.0, 1.0));

        let center = (lower + upper) / 2.0;
        let radius = (upper - lower) / 2.0;
        spec_lines.push(format!("{},{},{}", i, center, radius));
    }

    spec_lines.join("\n")
}

/// Select optimal domain based on property characteristics
///
/// - DeepZ: Fast, good for small epsilons
/// - DeepPoly: More precise, default choice
/// - RefinePoly: Most precise but slower
pub fn select_domain_for_property(spec: &TypedSpec, default_domain: EranDomain) -> EranDomain {
    // Extract epsilon to determine domain
    let mut epsilon = None;
    for prop in &spec.spec.properties {
        let expr = match prop {
            Property::Invariant(inv) => Some(&inv.body),
            Property::Theorem(thm) => Some(&thm.body),
            Property::Security(sec) => Some(&sec.body),
            Property::Probabilistic(prob) => Some(&prob.condition),
            _ => None,
        };
        if let Some(e) = expr {
            if let Some(eps) = extract_epsilon(e) {
                epsilon = Some(eps);
                break;
            }
        }
    }

    // Select domain based on epsilon
    if let Some(eps) = epsilon {
        if eps <= 0.001 {
            // Very tight bound - DeepZ is sufficient
            EranDomain::DeepZ
        } else if eps <= 0.1 {
            // Moderate bound - DeepPoly
            EranDomain::DeepPoly
        } else {
            // Large bound - need RefinePoly
            EranDomain::RefinePoly
        }
    } else {
        // Default to configured domain
        default_domain
    }
}
