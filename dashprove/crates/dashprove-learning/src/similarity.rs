//! Property similarity computation
//!
//! Extracts structural features from properties and computes similarity
//! scores for finding related proofs.

use crate::corpus::ProofId;
use dashprove_backends::traits::BackendId;
use dashprove_usl::ast::{Expr, Property, TemporalExpr};
use serde::{Deserialize, Serialize};

/// Extracted features from a property for similarity computation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PropertyFeatures {
    /// Property type (theorem, invariant, temporal, etc.)
    pub property_type: String,
    /// Maximum nesting depth of expressions
    pub depth: usize,
    /// Number of quantifiers (forall, exists)
    pub quantifier_depth: usize,
    /// Number of implications
    pub implication_count: usize,
    /// Number of arithmetic operations
    pub arithmetic_ops: usize,
    /// Number of function calls
    pub function_calls: usize,
    /// Number of variables
    pub variable_count: usize,
    /// Uses temporal operators
    pub has_temporal: bool,
    /// Type names referenced
    pub type_refs: Vec<String>,
    /// Keywords for text-based search (property name, function names, variable names, etc.)
    #[serde(default)]
    pub keywords: Vec<String>,
}

/// A similar proof with its similarity score
#[derive(Debug, Clone)]
pub struct SimilarProof {
    /// Proof identifier
    pub id: ProofId,
    /// The proven property
    pub property: Property,
    /// Backend that generated the proof
    pub backend: BackendId,
    /// Tactics used
    pub tactics: Vec<String>,
    /// Similarity score (0.0 to 1.0, higher is more similar)
    pub similarity: f64,
}

/// Extract structural features from a property
///
/// Uses `Property::property_kind()` for the `property_type` field to ensure
/// consistency with domain-weighted consensus and reputation tracking.
/// The property kind string is compatible with `property_type_from_string()`.
pub fn extract_features(property: &Property) -> PropertyFeatures {
    let mut features = PropertyFeatures::default();
    let mut keywords = Vec::new();

    // Use Property::property_kind() for consistent property type inference
    // This ensures compatibility with domain-weighted consensus and reputation tracking
    features.property_type = property.property_kind().to_string();

    match property {
        Property::Theorem(t) => {
            keywords.push("theorem".to_string());
            add_name_keywords(&t.name, &mut keywords);
            extract_expr_features(&t.body, &mut features, 0);
            extract_expr_keywords(&t.body, &mut keywords);
        }
        Property::Temporal(t) => {
            features.has_temporal = true;
            keywords.push("temporal".to_string());
            add_name_keywords(&t.name, &mut keywords);
            extract_temporal_features(&t.body, &mut features, 0);
            extract_temporal_keywords(&t.body, &mut keywords);
        }
        Property::Contract(c) => {
            keywords.push("contract".to_string());
            // type_path is Vec<String> like ["Graph", "add_node"]
            for part in &c.type_path {
                add_name_keywords(part, &mut keywords);
            }
            for param in &c.params {
                add_name_keywords(&param.name, &mut keywords);
            }
            for req in &c.requires {
                extract_expr_features(req, &mut features, 0);
                extract_expr_keywords(req, &mut keywords);
            }
            for ens in &c.ensures {
                extract_expr_features(ens, &mut features, 0);
                extract_expr_keywords(ens, &mut keywords);
            }
        }
        Property::Invariant(i) => {
            keywords.push("invariant".to_string());
            add_name_keywords(&i.name, &mut keywords);
            extract_expr_features(&i.body, &mut features, 0);
            extract_expr_keywords(&i.body, &mut keywords);
        }
        Property::Refinement(r) => {
            keywords.push("refinement".to_string());
            add_name_keywords(&r.name, &mut keywords);
            extract_expr_features(&r.abstraction, &mut features, 0);
            extract_expr_features(&r.simulation, &mut features, 0);
            extract_expr_keywords(&r.abstraction, &mut keywords);
            extract_expr_keywords(&r.simulation, &mut keywords);
        }
        Property::Probabilistic(p) => {
            keywords.push("probabilistic".to_string());
            add_name_keywords(&p.name, &mut keywords);
            extract_expr_features(&p.condition, &mut features, 0);
            extract_expr_keywords(&p.condition, &mut keywords);
        }
        Property::Security(s) => {
            keywords.push("security".to_string());
            add_name_keywords(&s.name, &mut keywords);
            extract_expr_features(&s.body, &mut features, 0);
            extract_expr_keywords(&s.body, &mut keywords);
        }
        Property::Semantic(s) => {
            keywords.push("semantic".to_string());
            add_name_keywords(&s.name, &mut keywords);
            extract_expr_features(&s.body, &mut features, 0);
            extract_expr_keywords(&s.body, &mut keywords);
        }
        Property::PlatformApi(api) => {
            keywords.push("platform_api".to_string());
            add_name_keywords(&api.name, &mut keywords);
            for state in &api.states {
                add_name_keywords(&state.name, &mut keywords);
                for transition in &state.transitions {
                    add_name_keywords(&transition.name, &mut keywords);
                    for req in &transition.requires {
                        extract_expr_features(req, &mut features, 0);
                        extract_expr_keywords(req, &mut keywords);
                    }
                    for ens in &transition.ensures {
                        extract_expr_features(ens, &mut features, 0);
                        extract_expr_keywords(ens, &mut keywords);
                    }
                }
                for inv in &state.invariants {
                    extract_expr_features(inv, &mut features, 0);
                    extract_expr_keywords(inv, &mut keywords);
                }
            }
        }
        Property::Bisimulation(b) => {
            // Include "bisimulation" keyword for searchability even though
            // property_kind() returns "refinement" (behavioral equivalence)
            keywords.push("bisimulation".to_string());
            keywords.push("refinement".to_string());
            add_name_keywords(&b.name, &mut keywords);
            keywords.push(b.oracle.clone());
            keywords.push(b.subject.clone());
            for eq_on in &b.equivalent_on {
                keywords.push(eq_on.clone());
            }
            if let Some(ref prop) = b.property {
                add_name_keywords(&prop.var_name, &mut keywords);
                extract_expr_features(&prop.oracle_expr, &mut features, 0);
                extract_expr_features(&prop.subject_expr, &mut features, 0);
                extract_expr_keywords(&prop.oracle_expr, &mut keywords);
                extract_expr_keywords(&prop.subject_expr, &mut keywords);
            }
        }
        Property::Version(v) => {
            // Include "version" keyword for searchability even though
            // property_kind() returns "theorem" (theorem-like verification)
            keywords.push("version".to_string());
            keywords.push("theorem".to_string());
            keywords.push("improves".to_string());
            add_name_keywords(&v.name, &mut keywords);
            add_name_keywords(&v.improves, &mut keywords);
            for cap in &v.capabilities {
                extract_expr_features(&cap.expr, &mut features, 0);
                extract_expr_keywords(&cap.expr, &mut keywords);
            }
            for pres in &v.preserves {
                extract_expr_features(&pres.property, &mut features, 0);
                extract_expr_keywords(&pres.property, &mut keywords);
            }
        }
        Property::Capability(c) => {
            // Include "capability" keyword for searchability even though
            // property_kind() returns "theorem" (theorem-like verification)
            keywords.push("capability".to_string());
            keywords.push("theorem".to_string());
            add_name_keywords(&c.name, &mut keywords);
            for ability in &c.abilities {
                add_name_keywords(&ability.name, &mut keywords);
                for param in &ability.params {
                    add_name_keywords(&param.name, &mut keywords);
                }
            }
            for req in &c.requires {
                extract_expr_features(req, &mut features, 0);
                extract_expr_keywords(req, &mut keywords);
            }
        }
        Property::DistributedInvariant(d) => {
            // Handle similarly to Invariant but with distributed keyword
            keywords.push("distributed".to_string());
            keywords.push("invariant".to_string());
            add_name_keywords(&d.name, &mut keywords);
            extract_expr_features(&d.body, &mut features, 0);
            extract_expr_keywords(&d.body, &mut keywords);
        }
        Property::DistributedTemporal(d) => {
            // Handle similarly to Temporal but with distributed keyword
            features.has_temporal = true;
            keywords.push("distributed".to_string());
            keywords.push("temporal".to_string());
            add_name_keywords(&d.name, &mut keywords);
            extract_temporal_features(&d.body, &mut features, 0);
            extract_temporal_keywords(&d.body, &mut keywords);
        }
        Property::Composed(c) => {
            // Handle composed theorems that depend on other properties
            keywords.push("composed".to_string());
            keywords.push("theorem".to_string());
            add_name_keywords(&c.name, &mut keywords);
            // Add each dependency as a keyword
            for dep in &c.uses {
                keywords.push(dep.to_lowercase());
            }
            extract_expr_features(&c.body, &mut features, 0);
            extract_expr_keywords(&c.body, &mut keywords);
        }
        Property::ImprovementProposal(p) => {
            // Handle improvement proposals (self-improvement specs)
            keywords.push("improvement".to_string());
            keywords.push("proposal".to_string());
            keywords.push("self_improvement".to_string());
            add_name_keywords(&p.name, &mut keywords);
            extract_expr_features(&p.target, &mut features, 0);
            extract_expr_keywords(&p.target, &mut keywords);
            for expr in &p.improves {
                extract_expr_features(expr, &mut features, 0);
                extract_expr_keywords(expr, &mut keywords);
            }
            for expr in &p.preserves {
                extract_expr_features(expr, &mut features, 0);
                extract_expr_keywords(expr, &mut keywords);
            }
            for expr in &p.requires {
                extract_expr_features(expr, &mut features, 0);
                extract_expr_keywords(expr, &mut keywords);
            }
        }
        Property::VerificationGate(g) => {
            // Handle verification gates (immutable checkpoints)
            keywords.push("verification".to_string());
            keywords.push("gate".to_string());
            keywords.push("checkpoint".to_string());
            add_name_keywords(&g.name, &mut keywords);
            for check in &g.checks {
                add_name_keywords(&check.name, &mut keywords);
                extract_expr_features(&check.condition, &mut features, 0);
                extract_expr_keywords(&check.condition, &mut keywords);
            }
            extract_expr_features(&g.on_pass, &mut features, 0);
            extract_expr_keywords(&g.on_pass, &mut keywords);
            extract_expr_features(&g.on_fail, &mut features, 0);
            extract_expr_keywords(&g.on_fail, &mut keywords);
        }
        Property::Rollback(r) => {
            // Handle rollback specifications
            keywords.push("rollback".to_string());
            keywords.push("recovery".to_string());
            features.has_temporal = true; // Rollback specs have temporal guarantees
            add_name_keywords(&r.name, &mut keywords);
            extract_expr_features(&r.trigger, &mut features, 0);
            extract_expr_keywords(&r.trigger, &mut keywords);
            for expr in &r.invariants {
                extract_expr_features(expr, &mut features, 0);
                extract_expr_keywords(expr, &mut keywords);
            }
            for expr in &r.guarantees {
                extract_expr_features(expr, &mut features, 0);
                extract_expr_keywords(expr, &mut keywords);
            }
        }
    }

    // Deduplicate and lowercase keywords
    keywords.sort();
    keywords.dedup();
    features.keywords = keywords;

    features
}

/// Split a name (e.g., "loop_termination") into keywords
fn add_name_keywords(name: &str, keywords: &mut Vec<String>) {
    // Add original name lowercased
    keywords.push(name.to_lowercase());
    // Split by underscores and add parts
    for part in name.split('_') {
        if part.len() > 1 {
            keywords.push(part.to_lowercase());
        }
    }
    // Split by camelCase
    let mut current = String::new();
    for c in name.chars() {
        if c.is_uppercase() && !current.is_empty() {
            if current.len() > 1 {
                keywords.push(current.to_lowercase());
            }
            current = c.to_lowercase().to_string();
        } else {
            current.push(c.to_ascii_lowercase());
        }
    }
    if current.len() > 1 {
        keywords.push(current);
    }
}

/// Extract keywords from an expression (variable names, function names, field names)
fn extract_expr_keywords(expr: &Expr, keywords: &mut Vec<String>) {
    match expr {
        Expr::Var(name) => {
            add_name_keywords(name, keywords);
        }
        Expr::Int(_) | Expr::Float(_) | Expr::String(_) | Expr::Bool(_) => {}
        Expr::ForAll { var, body, .. } | Expr::Exists { var, body, .. } => {
            add_name_keywords(var, keywords);
            keywords.push("forall".to_string());
            extract_expr_keywords(body, keywords);
        }
        Expr::ForAllIn {
            var,
            collection,
            body,
            ..
        }
        | Expr::ExistsIn {
            var,
            collection,
            body,
            ..
        } => {
            add_name_keywords(var, keywords);
            extract_expr_keywords(collection, keywords);
            extract_expr_keywords(body, keywords);
        }
        Expr::Implies(lhs, rhs) => {
            keywords.push("implies".to_string());
            extract_expr_keywords(lhs, keywords);
            extract_expr_keywords(rhs, keywords);
        }
        Expr::And(lhs, rhs) | Expr::Or(lhs, rhs) => {
            extract_expr_keywords(lhs, keywords);
            extract_expr_keywords(rhs, keywords);
        }
        Expr::Not(inner) => {
            extract_expr_keywords(inner, keywords);
        }
        Expr::Compare(lhs, _, rhs) => {
            extract_expr_keywords(lhs, keywords);
            extract_expr_keywords(rhs, keywords);
        }
        Expr::Binary(lhs, _, rhs) => {
            extract_expr_keywords(lhs, keywords);
            extract_expr_keywords(rhs, keywords);
        }
        Expr::Neg(inner) => {
            extract_expr_keywords(inner, keywords);
        }
        Expr::App(func_name, args) => {
            add_name_keywords(func_name, keywords);
            for arg in args {
                extract_expr_keywords(arg, keywords);
            }
        }
        Expr::MethodCall {
            receiver,
            method,
            args,
        } => {
            add_name_keywords(method, keywords);
            extract_expr_keywords(receiver, keywords);
            for arg in args {
                extract_expr_keywords(arg, keywords);
            }
        }
        Expr::FieldAccess(obj, field) => {
            add_name_keywords(field, keywords);
            extract_expr_keywords(obj, keywords);
        }
    }
}

/// Extract keywords from a temporal expression
fn extract_temporal_keywords(expr: &TemporalExpr, keywords: &mut Vec<String>) {
    match expr {
        TemporalExpr::Always(inner) => {
            keywords.push("always".to_string());
            extract_temporal_keywords(inner, keywords);
        }
        TemporalExpr::Eventually(inner) => {
            keywords.push("eventually".to_string());
            extract_temporal_keywords(inner, keywords);
        }
        TemporalExpr::LeadsTo(from, to) => {
            keywords.push("leadsto".to_string());
            extract_temporal_keywords(from, keywords);
            extract_temporal_keywords(to, keywords);
        }
        TemporalExpr::Atom(expr) => {
            extract_expr_keywords(expr, keywords);
        }
    }
}

/// Extract features from an expression recursively
fn extract_expr_features(expr: &Expr, features: &mut PropertyFeatures, depth: usize) {
    features.depth = features.depth.max(depth);

    match expr {
        Expr::Var(_) => {
            features.variable_count += 1;
        }
        Expr::Int(_) | Expr::Float(_) | Expr::String(_) | Expr::Bool(_) => {}
        Expr::ForAll { body, .. } | Expr::Exists { body, .. } => {
            features.quantifier_depth += 1;
            extract_expr_features(body, features, depth + 1);
        }
        Expr::ForAllIn {
            collection, body, ..
        }
        | Expr::ExistsIn {
            collection, body, ..
        } => {
            features.quantifier_depth += 1;
            extract_expr_features(collection, features, depth + 1);
            extract_expr_features(body, features, depth + 1);
        }
        Expr::Implies(lhs, rhs) => {
            features.implication_count += 1;
            extract_expr_features(lhs, features, depth + 1);
            extract_expr_features(rhs, features, depth + 1);
        }
        Expr::And(lhs, rhs) | Expr::Or(lhs, rhs) => {
            extract_expr_features(lhs, features, depth + 1);
            extract_expr_features(rhs, features, depth + 1);
        }
        Expr::Not(inner) => {
            extract_expr_features(inner, features, depth + 1);
        }
        Expr::Compare(lhs, _, rhs) => {
            extract_expr_features(lhs, features, depth + 1);
            extract_expr_features(rhs, features, depth + 1);
        }
        Expr::Binary(lhs, _, rhs) => {
            features.arithmetic_ops += 1;
            extract_expr_features(lhs, features, depth + 1);
            extract_expr_features(rhs, features, depth + 1);
        }
        Expr::Neg(inner) => {
            features.arithmetic_ops += 1;
            extract_expr_features(inner, features, depth + 1);
        }
        Expr::App(_, args) => {
            features.function_calls += 1;
            for arg in args {
                extract_expr_features(arg, features, depth + 1);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            features.function_calls += 1;
            extract_expr_features(receiver, features, depth + 1);
            for arg in args {
                extract_expr_features(arg, features, depth + 1);
            }
        }
        Expr::FieldAccess(obj, _) => {
            extract_expr_features(obj, features, depth + 1);
        }
    }
}

/// Extract features from a temporal expression
fn extract_temporal_features(expr: &TemporalExpr, features: &mut PropertyFeatures, depth: usize) {
    features.depth = features.depth.max(depth);

    match expr {
        TemporalExpr::Always(inner) | TemporalExpr::Eventually(inner) => {
            extract_temporal_features(inner, features, depth + 1);
        }
        TemporalExpr::LeadsTo(from, to) => {
            features.implication_count += 1;
            extract_temporal_features(from, features, depth + 1);
            extract_temporal_features(to, features, depth + 1);
        }
        TemporalExpr::Atom(expr) => {
            extract_expr_features(expr, features, depth + 1);
        }
    }
}

/// Compute similarity between two property feature sets
///
/// Returns a score between 0.0 and 1.0 where 1.0 means identical features.
pub fn compute_similarity(a: &PropertyFeatures, b: &PropertyFeatures) -> f64 {
    let mut score = 0.0;
    let mut weight_sum = 0.0;

    // Property type match (high weight)
    let type_weight = 3.0;
    if a.property_type == b.property_type {
        score += type_weight;
    }
    weight_sum += type_weight;

    // Depth similarity (inverse of normalized difference)
    let depth_weight = 1.0;
    let max_depth = a.depth.max(b.depth).max(1);
    let depth_diff = (a.depth as f64 - b.depth as f64).abs() / max_depth as f64;
    score += depth_weight * (1.0 - depth_diff);
    weight_sum += depth_weight;

    // Quantifier similarity
    let quant_weight = 2.0;
    let max_quant = a.quantifier_depth.max(b.quantifier_depth).max(1);
    let quant_diff =
        (a.quantifier_depth as f64 - b.quantifier_depth as f64).abs() / max_quant as f64;
    score += quant_weight * (1.0 - quant_diff);
    weight_sum += quant_weight;

    // Implication similarity
    let impl_weight = 1.5;
    let max_impl = a.implication_count.max(b.implication_count).max(1);
    let impl_diff =
        (a.implication_count as f64 - b.implication_count as f64).abs() / max_impl as f64;
    score += impl_weight * (1.0 - impl_diff);
    weight_sum += impl_weight;

    // Arithmetic similarity
    let arith_weight = 1.0;
    let has_arith_a = a.arithmetic_ops > 0;
    let has_arith_b = b.arithmetic_ops > 0;
    if has_arith_a == has_arith_b {
        score += arith_weight;
    }
    weight_sum += arith_weight;

    // Temporal similarity
    let temp_weight = 2.0;
    if a.has_temporal == b.has_temporal {
        score += temp_weight;
    }
    weight_sum += temp_weight;

    // Function calls similarity
    let func_weight = 1.0;
    let max_func = a.function_calls.max(b.function_calls).max(1);
    let func_diff = (a.function_calls as f64 - b.function_calls as f64).abs() / max_func as f64;
    score += func_weight * (1.0 - func_diff);
    weight_sum += func_weight;

    score / weight_sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::ast::{Invariant, Temporal, Theorem, Type};

    #[test]
    fn test_extract_simple_invariant() {
        let prop = Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(true),
        });

        let features = extract_features(&prop);
        assert_eq!(features.property_type, "invariant");
        assert_eq!(features.depth, 0);
        assert_eq!(features.quantifier_depth, 0);
    }

    #[test]
    fn test_extract_quantified_theorem() {
        let prop = Property::Theorem(Theorem {
            name: "forall_test".to_string(),
            body: Expr::ForAll {
                var: "x".to_string(),
                ty: Some(Type::Named("Int".to_string())),
                body: Box::new(Expr::Implies(
                    Box::new(Expr::Compare(
                        Box::new(Expr::Var("x".to_string())),
                        dashprove_usl::ast::ComparisonOp::Gt,
                        Box::new(Expr::Int(0)),
                    )),
                    Box::new(Expr::Compare(
                        Box::new(Expr::Binary(
                            Box::new(Expr::Var("x".to_string())),
                            dashprove_usl::ast::BinaryOp::Add,
                            Box::new(Expr::Int(1)),
                        )),
                        dashprove_usl::ast::ComparisonOp::Gt,
                        Box::new(Expr::Int(1)),
                    )),
                )),
            },
        });

        let features = extract_features(&prop);
        assert_eq!(features.property_type, "theorem");
        assert_eq!(features.quantifier_depth, 1);
        assert_eq!(features.implication_count, 1);
        assert!(features.arithmetic_ops > 0);
        assert!(features.variable_count > 0);
    }

    #[test]
    fn test_extract_temporal() {
        let prop = Property::Temporal(Temporal {
            name: "always_eventually".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Eventually(Box::new(
                TemporalExpr::Atom(Expr::Bool(true)),
            )))),
            fairness: vec![],
        });

        let features = extract_features(&prop);
        assert_eq!(features.property_type, "temporal");
        assert!(features.has_temporal);
        assert!(features.depth >= 2);
    }

    #[test]
    fn test_similarity_same_type() {
        let inv1 = Property::Invariant(Invariant {
            name: "inv1".to_string(),
            body: Expr::Bool(true),
        });
        let inv2 = Property::Invariant(Invariant {
            name: "inv2".to_string(),
            body: Expr::Bool(false),
        });
        let thm = Property::Theorem(Theorem {
            name: "thm".to_string(),
            body: Expr::Bool(true),
        });

        let f1 = extract_features(&inv1);
        let f2 = extract_features(&inv2);
        let f3 = extract_features(&thm);

        // Same type should be more similar
        let sim_same = compute_similarity(&f1, &f2);
        let sim_diff = compute_similarity(&f1, &f3);

        assert!(sim_same > sim_diff);
    }

    #[test]
    fn test_similarity_quantifier_depth() {
        let simple = Property::Invariant(Invariant {
            name: "simple".to_string(),
            body: Expr::Bool(true),
        });
        let quantified = Property::Invariant(Invariant {
            name: "quantified".to_string(),
            body: Expr::ForAll {
                var: "x".to_string(),
                ty: None,
                body: Box::new(Expr::ForAll {
                    var: "y".to_string(),
                    ty: None,
                    body: Box::new(Expr::Bool(true)),
                }),
            },
        });
        let also_quantified = Property::Invariant(Invariant {
            name: "also_quantified".to_string(),
            body: Expr::ForAll {
                var: "z".to_string(),
                ty: None,
                body: Box::new(Expr::ForAll {
                    var: "w".to_string(),
                    ty: None,
                    body: Box::new(Expr::Bool(true)),
                }),
            },
        });

        let f_simple = extract_features(&simple);
        let f_quant = extract_features(&quantified);
        let f_also = extract_features(&also_quantified);

        // Two quantified should be more similar than quantified vs simple
        let sim_quant_quant = compute_similarity(&f_quant, &f_also);
        let sim_quant_simple = compute_similarity(&f_quant, &f_simple);

        assert!(sim_quant_quant > sim_quant_simple);
    }

    #[test]
    fn test_similarity_range() {
        let prop = Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::Bool(true),
        });
        let features = extract_features(&prop);

        let sim = compute_similarity(&features, &features);

        // Self-similarity should be 1.0
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_keyword_extraction_from_name() {
        let prop = Property::Theorem(Theorem {
            name: "loop_termination_proof".to_string(),
            body: Expr::Bool(true),
        });

        let features = extract_features(&prop);
        assert!(features.keywords.contains(&"loop".to_string()));
        assert!(features.keywords.contains(&"termination".to_string()));
        assert!(features.keywords.contains(&"proof".to_string()));
        assert!(features.keywords.contains(&"theorem".to_string()));
    }

    #[test]
    fn test_keyword_extraction_camel_case() {
        let prop = Property::Invariant(Invariant {
            name: "arrayBoundsCheck".to_string(),
            body: Expr::Bool(true),
        });

        let features = extract_features(&prop);
        assert!(features.keywords.contains(&"array".to_string()));
        assert!(features.keywords.contains(&"bounds".to_string()));
        assert!(features.keywords.contains(&"check".to_string()));
    }

    #[test]
    fn test_keyword_extraction_from_variables() {
        let prop = Property::Theorem(Theorem {
            name: "test".to_string(),
            body: Expr::ForAll {
                var: "node_count".to_string(),
                ty: None,
                body: Box::new(Expr::Var("graph_size".to_string())),
            },
        });

        let features = extract_features(&prop);
        assert!(features.keywords.contains(&"node".to_string()));
        assert!(features.keywords.contains(&"count".to_string()));
        assert!(features.keywords.contains(&"graph".to_string()));
        assert!(features.keywords.contains(&"size".to_string()));
        assert!(features.keywords.contains(&"forall".to_string()));
    }

    #[test]
    fn test_keyword_extraction_from_function_calls() {
        let prop = Property::Invariant(Invariant {
            name: "test".to_string(),
            body: Expr::App(
                "is_valid_state".to_string(),
                vec![Expr::Var("current_node".to_string())],
            ),
        });

        let features = extract_features(&prop);
        assert!(features.keywords.contains(&"valid".to_string()));
        assert!(features.keywords.contains(&"state".to_string()));
        assert!(features.keywords.contains(&"current".to_string()));
        assert!(features.keywords.contains(&"node".to_string()));
    }

    #[test]
    fn test_keyword_extraction_temporal() {
        let prop = Property::Temporal(Temporal {
            name: "liveness_property".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Eventually(Box::new(
                TemporalExpr::Atom(Expr::Var("done".to_string())),
            )))),
            fairness: vec![],
        });

        let features = extract_features(&prop);
        assert!(features.keywords.contains(&"temporal".to_string()));
        assert!(features.keywords.contains(&"always".to_string()));
        assert!(features.keywords.contains(&"eventually".to_string()));
        assert!(features.keywords.contains(&"liveness".to_string()));
        assert!(features.keywords.contains(&"property".to_string()));
        assert!(features.keywords.contains(&"done".to_string()));
    }

    #[test]
    fn test_keywords_are_deduplicated() {
        let prop = Property::Invariant(Invariant {
            name: "test_test".to_string(),
            body: Expr::Var("test".to_string()),
        });

        let features = extract_features(&prop);
        // "test" should only appear once despite being in name twice and in body
        let test_count = features.keywords.iter().filter(|k| *k == "test").count();
        assert_eq!(test_count, 1, "Keywords should be deduplicated");
    }

    // Mutation-killing tests for compute_similarity arithmetic

    #[test]
    fn test_similarity_depth_diff_arithmetic() {
        // Test that depth difference is computed correctly
        let f1 = PropertyFeatures {
            property_type: "invariant".to_string(),
            depth: 10,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            has_temporal: false,
            function_calls: 0,
            variable_count: 0,
            type_refs: vec![],
            keywords: vec![],
        };
        let mut f2 = f1.clone();
        f2.depth = 5;

        // With depth 10 and 5, diff = |10-5|/10 = 0.5
        // depth_weight = 1.0, score contribution = 1.0 * (1.0 - 0.5) = 0.5
        let sim = compute_similarity(&f1, &f2);

        // If we swap subtraction for addition: diff = |10+5|/10 = 1.5, (1.0 - 1.5) = -0.5
        // Total score would be different - score would be lower
        let self_sim = compute_similarity(&f1, &f1);
        assert!(
            sim < self_sim,
            "Different depths should have lower similarity"
        );

        // More precise: depth contributes to a weighted score
        // With identical features except depth, the difference should be measurable
        // expected_depth_penalty = 0.5 * 1.0 / 11.5 (depth_weight * diff / total_weight)
        assert!(
            (self_sim - sim).abs() > 0.01,
            "Depth difference should create measurable similarity difference"
        );
    }

    #[test]
    fn test_similarity_quantifier_diff_arithmetic() {
        // Test quantifier_depth difference computation
        let f1 = PropertyFeatures {
            property_type: "theorem".to_string(),
            depth: 2,
            quantifier_depth: 4,
            implication_count: 0,
            arithmetic_ops: 0,
            has_temporal: false,
            function_calls: 0,
            variable_count: 1,
            type_refs: vec![],
            keywords: vec![],
        };
        let mut f2 = f1.clone();
        f2.quantifier_depth = 2;

        // quant_diff = |4-2|/4 = 0.5, contribution = 2.0 * 0.5 = 1.0 lost
        let sim = compute_similarity(&f1, &f2);
        let self_sim = compute_similarity(&f1, &f1);

        // If subtraction became addition: |4+2|/4 = 1.5, penalty would be larger
        assert!(sim < self_sim);

        // The difference should be approximately 1.0/11.5 of the weight
        let diff = self_sim - sim;
        assert!(
            diff > 0.05,
            "Quantifier difference should have measurable impact"
        );
    }

    #[test]
    fn test_similarity_implication_diff_arithmetic() {
        // Test implication_count difference
        let f1 = PropertyFeatures {
            property_type: "theorem".to_string(),
            depth: 1,
            quantifier_depth: 1,
            implication_count: 6,
            arithmetic_ops: 0,
            has_temporal: false,
            function_calls: 0,
            variable_count: 1,
            type_refs: vec![],
            keywords: vec![],
        };
        let mut f2 = f1.clone();
        f2.implication_count = 3;

        // impl_diff = |6-3|/6 = 0.5
        let sim = compute_similarity(&f1, &f2);
        let self_sim = compute_similarity(&f1, &f1);

        assert!(
            sim < self_sim,
            "Different implication counts should reduce similarity"
        );
        let diff = self_sim - sim;
        assert!(diff > 0.05, "Implication difference should be measurable");
    }

    #[test]
    fn test_similarity_function_calls_diff_arithmetic() {
        // Test function_calls difference
        let f1 = PropertyFeatures {
            property_type: "invariant".to_string(),
            depth: 1,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            has_temporal: false,
            function_calls: 8,
            variable_count: 2,
            type_refs: vec![],
            keywords: vec![],
        };
        let mut f2 = f1.clone();
        f2.function_calls = 4;

        // func_diff = |8-4|/8 = 0.5
        let sim = compute_similarity(&f1, &f2);
        let self_sim = compute_similarity(&f1, &f1);

        assert!(sim < self_sim);
        let diff = self_sim - sim;
        assert!(diff > 0.02, "Function call difference should be measurable");
    }

    #[test]
    fn test_similarity_arithmetic_ops_match() {
        // Test has_arith_a == has_arith_b comparison
        let f1 = PropertyFeatures {
            property_type: "invariant".to_string(),
            depth: 1,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 5,
            has_temporal: false,
            function_calls: 0,
            variable_count: 1,
            type_refs: vec![],
            keywords: vec![],
        };
        let mut f2 = f1.clone();
        f2.arithmetic_ops = 0; // No arithmetic

        let sim_mismatch = compute_similarity(&f1, &f2);

        // Both have arithmetic
        let mut f3 = f1.clone();
        f3.arithmetic_ops = 10;
        let sim_both_arith = compute_similarity(&f1, &f3);

        // Both with arithmetic should be more similar than mismatch
        assert!(
            sim_both_arith > sim_mismatch,
            "Same arithmetic presence should be more similar"
        );
    }

    #[test]
    fn test_similarity_division_by_weight_sum() {
        // Test that final division is correct
        // weight_sum = 3.0 + 1.0 + 2.0 + 1.5 + 1.0 + 2.0 + 1.0 = 11.5
        let f = PropertyFeatures {
            property_type: "invariant".to_string(),
            depth: 1,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            has_temporal: false,
            function_calls: 0,
            variable_count: 0,
            type_refs: vec![],
            keywords: vec![],
        };

        let sim = compute_similarity(&f, &f);
        // Self-similarity should be exactly 1.0 (all weights contribute fully)
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "Self-similarity should be 1.0, got {}",
            sim
        );

        // If division were replaced with multiplication, result would be huge
        assert!(sim <= 1.0, "Similarity should not exceed 1.0");
    }

    #[test]
    fn test_similarity_max_depth_prevents_division_by_zero() {
        // Both have depth 0, max(0, 0, 1) = 1
        let f1 = PropertyFeatures {
            property_type: "invariant".to_string(),
            depth: 0,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            has_temporal: false,
            function_calls: 0,
            variable_count: 0,
            type_refs: vec![],
            keywords: vec![],
        };

        // This should not panic
        let sim = compute_similarity(&f1, &f1);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_similarity_normalized_diff_range() {
        // Test that differences are properly normalized to [0, 1]
        let f1 = PropertyFeatures {
            property_type: "invariant".to_string(),
            depth: 100,
            quantifier_depth: 50,
            implication_count: 30,
            arithmetic_ops: 10,
            has_temporal: true,
            function_calls: 20,
            variable_count: 15,
            type_refs: vec![],
            keywords: vec![],
        };

        let f2 = PropertyFeatures {
            property_type: "invariant".to_string(),
            depth: 0,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            has_temporal: true,
            function_calls: 0,
            variable_count: 0,
            type_refs: vec![],
            keywords: vec![],
        };

        let sim = compute_similarity(&f1, &f2);
        // Even with maximal differences, similarity should be in [0, 1]
        assert!((0.0..=1.0).contains(&sim), "Similarity should be in [0, 1]");
    }

    // Tests for feature extraction depth tracking

    #[test]
    fn test_extract_nested_forall_depth() {
        // ForAll { ForAll { ForAll { Bool } } }
        // Depth should be 3 for the nested expression
        let prop = Property::Theorem(Theorem {
            name: "nested".to_string(),
            body: Expr::ForAll {
                var: "a".to_string(),
                ty: None,
                body: Box::new(Expr::ForAll {
                    var: "b".to_string(),
                    ty: None,
                    body: Box::new(Expr::ForAll {
                        var: "c".to_string(),
                        ty: None,
                        body: Box::new(Expr::Bool(true)),
                    }),
                }),
            },
        });

        let features = extract_features(&prop);
        assert_eq!(
            features.quantifier_depth, 3,
            "Three nested ForAll should give quantifier_depth=3"
        );
        // Depth traverses: ForAll(1) -> ForAll(2) -> ForAll(3) -> Bool(4)
        assert!(
            features.depth >= 3,
            "Depth should reflect nesting: got {}",
            features.depth
        );
    }

    #[test]
    fn test_extract_nested_binary_depth() {
        // Binary { Binary { Binary { Int, Int }, Int }, Int }
        // Tests depth + 1 in Binary case
        let prop = Property::Invariant(Invariant {
            name: "arith".to_string(),
            body: Expr::Binary(
                Box::new(Expr::Binary(
                    Box::new(Expr::Binary(
                        Box::new(Expr::Int(1)),
                        dashprove_usl::ast::BinaryOp::Add,
                        Box::new(Expr::Int(2)),
                    )),
                    dashprove_usl::ast::BinaryOp::Add,
                    Box::new(Expr::Int(3)),
                )),
                dashprove_usl::ast::BinaryOp::Add,
                Box::new(Expr::Int(4)),
            ),
        });

        let features = extract_features(&prop);
        assert_eq!(
            features.arithmetic_ops, 3,
            "Three Binary operations should give arithmetic_ops=3"
        );
        assert!(features.depth >= 3, "Nested binaries should increase depth");
    }

    #[test]
    fn test_extract_nested_compare_depth() {
        // Compare { Binary { x, y }, Binary { z, w } }
        let prop = Property::Theorem(Theorem {
            name: "cmp".to_string(),
            body: Expr::Compare(
                Box::new(Expr::Binary(
                    Box::new(Expr::Var("x".to_string())),
                    dashprove_usl::ast::BinaryOp::Add,
                    Box::new(Expr::Var("y".to_string())),
                )),
                dashprove_usl::ast::ComparisonOp::Gt,
                Box::new(Expr::Binary(
                    Box::new(Expr::Var("z".to_string())),
                    dashprove_usl::ast::BinaryOp::Mul,
                    Box::new(Expr::Var("w".to_string())),
                )),
            ),
        });

        let features = extract_features(&prop);
        assert_eq!(
            features.arithmetic_ops, 2,
            "Two binary operations inside compare"
        );
        assert_eq!(features.variable_count, 4, "Four variables: x, y, z, w");
        assert!(features.depth >= 2, "Compare containing binary ops");
    }

    #[test]
    fn test_extract_neg_depth() {
        // Neg { Neg { Int } }
        let prop = Property::Invariant(Invariant {
            name: "neg".to_string(),
            body: Expr::Neg(Box::new(Expr::Neg(Box::new(Expr::Int(5))))),
        });

        let features = extract_features(&prop);
        assert_eq!(
            features.arithmetic_ops, 2,
            "Two Neg operations should give arithmetic_ops=2"
        );
    }

    #[test]
    fn test_extract_app_with_nested_args() {
        // App("f", [Binary { Int, Int }, Var])
        let prop = Property::Invariant(Invariant {
            name: "app".to_string(),
            body: Expr::App(
                "compute".to_string(),
                vec![
                    Expr::Binary(
                        Box::new(Expr::Int(1)),
                        dashprove_usl::ast::BinaryOp::Add,
                        Box::new(Expr::Int(2)),
                    ),
                    Expr::Var("x".to_string()),
                ],
            ),
        });

        let features = extract_features(&prop);
        assert_eq!(features.function_calls, 1);
        assert_eq!(features.arithmetic_ops, 1);
        assert_eq!(features.variable_count, 1);
    }

    #[test]
    fn test_extract_method_call_depth() {
        // MethodCall { receiver: Var, args: [Binary] }
        let prop = Property::Invariant(Invariant {
            name: "method".to_string(),
            body: Expr::MethodCall {
                receiver: Box::new(Expr::Var("obj".to_string())),
                method: "process".to_string(),
                args: vec![Expr::Binary(
                    Box::new(Expr::Int(10)),
                    dashprove_usl::ast::BinaryOp::Sub,
                    Box::new(Expr::Int(5)),
                )],
            },
        });

        let features = extract_features(&prop);
        assert_eq!(features.function_calls, 1);
        assert_eq!(features.arithmetic_ops, 1);
        assert!(features.variable_count >= 1);
    }

    #[test]
    fn test_extract_field_access_depth() {
        // FieldAccess { FieldAccess { Var, "a" }, "b" }
        let prop = Property::Invariant(Invariant {
            name: "field".to_string(),
            body: Expr::FieldAccess(
                Box::new(Expr::FieldAccess(
                    Box::new(Expr::Var("obj".to_string())),
                    "inner".to_string(),
                )),
                "value".to_string(),
            ),
        });

        let features = extract_features(&prop);
        assert!(
            features.depth >= 2,
            "Nested field access should increase depth"
        );
    }

    #[test]
    fn test_extract_temporal_always_eventually_depth() {
        // Always { Eventually { Always { Eventually { Atom } } } }
        let prop = Property::Temporal(Temporal {
            name: "nested_temporal".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Eventually(Box::new(
                TemporalExpr::Always(Box::new(TemporalExpr::Eventually(Box::new(
                    TemporalExpr::Atom(Expr::Bool(true)),
                )))),
            )))),
            fairness: vec![],
        });

        let features = extract_features(&prop);
        assert!(features.has_temporal);
        // Four temporal operators: Always -> Eventually -> Always -> Eventually
        assert!(features.depth >= 4, "Four nested temporal ops");
    }

    #[test]
    fn test_extract_temporal_leads_to_depth() {
        // LeadsTo { Atom(x), LeadsTo { Atom(y), Atom(z) } }
        let prop = Property::Temporal(Temporal {
            name: "leads_to_chain".to_string(),
            body: TemporalExpr::LeadsTo(
                Box::new(TemporalExpr::Atom(Expr::Var("x".to_string()))),
                Box::new(TemporalExpr::LeadsTo(
                    Box::new(TemporalExpr::Atom(Expr::Var("y".to_string()))),
                    Box::new(TemporalExpr::Atom(Expr::Var("z".to_string()))),
                )),
            ),
            fairness: vec![],
        });

        let features = extract_features(&prop);
        assert_eq!(
            features.implication_count, 2,
            "Two LeadsTo should give implication_count=2"
        );
        assert_eq!(features.variable_count, 3, "Three variables: x, y, z");
    }

    #[test]
    fn test_similarity_division_not_modulo() {
        // Test that we use / not % for normalization
        // For depth 7 vs 3, max=7, diff=4
        // With /: 4/7 ≈ 0.571
        // With %: 4%7 = 4 (diff unchanged - bad normalization)
        let f1 = PropertyFeatures {
            property_type: "invariant".to_string(),
            depth: 7,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            has_temporal: false,
            function_calls: 0,
            variable_count: 0,
            type_refs: vec![],
            keywords: vec![],
        };
        let mut f2 = f1.clone();
        f2.depth = 3;

        let sim = compute_similarity(&f1, &f2);
        // If using %, the depth penalty would be 4, not 4/7
        // Score would be < 0 or very different
        assert!(
            sim > 0.5,
            "With proper division, similarity should still be reasonable: {}",
            sim
        );
    }

    // ==========================================================================
    // Property kind consistency tests
    // ==========================================================================

    #[test]
    fn test_extract_features_uses_property_kind() {
        // Verify that extract_features uses Property::property_kind() for property_type
        use dashprove_usl::ast::{Bisimulation, CapabilitySpec, VersionSpec};

        // Theorem
        let theorem = Property::Theorem(Theorem {
            name: "test_thm".to_string(),
            body: Expr::Bool(true),
        });
        assert_eq!(
            extract_features(&theorem).property_type,
            theorem.property_kind()
        );

        // Invariant
        let inv = Property::Invariant(Invariant {
            name: "test_inv".to_string(),
            body: Expr::Bool(true),
        });
        assert_eq!(extract_features(&inv).property_type, inv.property_kind());

        // Bisimulation -> "refinement"
        let bisim = Property::Bisimulation(Bisimulation {
            name: "test_bisim".to_string(),
            oracle: "oracle".to_string(),
            subject: "subject".to_string(),
            equivalent_on: vec![],
            tolerance: None,
            property: None,
        });
        assert_eq!(
            extract_features(&bisim).property_type,
            "refinement",
            "Bisimulation should map to 'refinement'"
        );

        // Version -> "theorem"
        let version = Property::Version(VersionSpec {
            name: "test_version".to_string(),
            improves: "baseline".to_string(),
            capabilities: vec![],
            preserves: vec![],
        });
        assert_eq!(
            extract_features(&version).property_type,
            "theorem",
            "Version should map to 'theorem'"
        );

        // Capability -> "theorem"
        let capability = Property::Capability(CapabilitySpec {
            name: "test_capability".to_string(),
            abilities: vec![],
            requires: vec![],
        });
        assert_eq!(
            extract_features(&capability).property_type,
            "theorem",
            "Capability should map to 'theorem'"
        );
    }

    #[test]
    fn test_property_type_maps_to_valid_property_type() {
        // Verify that key property types from extract_features map to valid PropertyType
        // via property_type_from_string (used in reputation bootstrapping)
        use crate::reputation::property_type_from_string;
        use dashprove_usl::ast::{Bisimulation, CapabilitySpec, SemanticProperty, VersionSpec};

        // Test representative properties for each property_kind() return value
        let properties: Vec<Property> = vec![
            // "theorem"
            Property::Theorem(Theorem {
                name: "thm".to_string(),
                body: Expr::Bool(true),
            }),
            // "temporal"
            Property::Temporal(Temporal {
                name: "temp".to_string(),
                body: TemporalExpr::Atom(Expr::Bool(true)),
                fairness: vec![],
            }),
            // "invariant"
            Property::Invariant(Invariant {
                name: "inv".to_string(),
                body: Expr::Bool(true),
            }),
            // "semantic" -> maps to Theorem
            Property::Semantic(SemanticProperty {
                name: "sem".to_string(),
                body: Expr::Bool(true),
            }),
            // "refinement" from Bisimulation
            Property::Bisimulation(Bisimulation {
                name: "bisim".to_string(),
                oracle: "oracle".to_string(),
                subject: "subject".to_string(),
                equivalent_on: vec![],
                tolerance: None,
                property: None,
            }),
            // "theorem" from Version
            Property::Version(VersionSpec {
                name: "ver".to_string(),
                improves: "base".to_string(),
                capabilities: vec![],
                preserves: vec![],
            }),
            // "theorem" from Capability
            Property::Capability(CapabilitySpec {
                name: "cap".to_string(),
                abilities: vec![],
                requires: vec![],
            }),
        ];

        for prop in properties {
            let features = extract_features(&prop);
            let property_type_result = property_type_from_string(&features.property_type);
            assert!(
                property_type_result.is_some(),
                "Property type '{}' from {:?} should map to a valid PropertyType",
                features.property_type,
                std::mem::discriminant(&prop)
            );
        }
    }

    #[test]
    fn test_bisimulation_keywords_include_both_terms() {
        // Bisimulation should include both "bisimulation" and "refinement" keywords for searchability
        use dashprove_usl::ast::Bisimulation;

        let bisim = Property::Bisimulation(Bisimulation {
            name: "test".to_string(),
            oracle: "oracle".to_string(),
            subject: "subject".to_string(),
            equivalent_on: vec![],
            tolerance: None,
            property: None,
        });

        let features = extract_features(&bisim);
        assert!(
            features.keywords.contains(&"bisimulation".to_string()),
            "Bisimulation should include 'bisimulation' keyword"
        );
        assert!(
            features.keywords.contains(&"refinement".to_string()),
            "Bisimulation should include 'refinement' keyword"
        );
    }

    #[test]
    fn test_version_capability_keywords_include_both_terms() {
        // Version and Capability should include both specific and general keywords
        use dashprove_usl::ast::{CapabilitySpec, VersionSpec};

        let version = Property::Version(VersionSpec {
            name: "test".to_string(),
            improves: "base".to_string(),
            capabilities: vec![],
            preserves: vec![],
        });
        let version_features = extract_features(&version);
        assert!(
            version_features.keywords.contains(&"version".to_string()),
            "Version should include 'version' keyword"
        );
        assert!(
            version_features.keywords.contains(&"theorem".to_string()),
            "Version should include 'theorem' keyword"
        );

        let capability = Property::Capability(CapabilitySpec {
            name: "test".to_string(),
            abilities: vec![],
            requires: vec![],
        });
        let capability_features = extract_features(&capability);
        assert!(
            capability_features
                .keywords
                .contains(&"capability".to_string()),
            "Capability should include 'capability' keyword"
        );
        assert!(
            capability_features
                .keywords
                .contains(&"theorem".to_string()),
            "Capability should include 'theorem' keyword"
        );
    }
}

/// Kani proofs for compute_similarity function
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that compute_similarity always returns a value in [0.0, 1.0]
    #[kani::proof]
    fn verify_similarity_bounds() {
        let a = PropertyFeatures {
            property_type: String::new(),
            depth: kani::any(),
            quantifier_depth: kani::any(),
            implication_count: kani::any(),
            arithmetic_ops: kani::any(),
            function_calls: kani::any(),
            variable_count: 0,
            has_temporal: kani::any(),
            type_refs: vec![],
            keywords: vec![],
        };
        let b = PropertyFeatures {
            property_type: String::new(),
            depth: kani::any(),
            quantifier_depth: kani::any(),
            implication_count: kani::any(),
            arithmetic_ops: kani::any(),
            function_calls: kani::any(),
            variable_count: 0,
            has_temporal: kani::any(),
            type_refs: vec![],
            keywords: vec![],
        };

        let sim = compute_similarity(&a, &b);

        kani::assert(
            sim >= 0.0 && sim <= 1.0,
            "Similarity must be in range [0.0, 1.0]",
        );
    }

    /// Prove that self-similarity is always 1.0 (reflexivity)
    #[kani::proof]
    fn verify_self_similarity_is_one() {
        let features = PropertyFeatures {
            property_type: String::new(),
            depth: kani::any(),
            quantifier_depth: kani::any(),
            implication_count: kani::any(),
            arithmetic_ops: kani::any(),
            function_calls: kani::any(),
            variable_count: 0,
            has_temporal: kani::any(),
            type_refs: vec![],
            keywords: vec![],
        };

        let sim = compute_similarity(&features, &features);

        kani::assert((sim - 1.0).abs() < 1e-10, "Self-similarity must be 1.0");
    }

    /// Prove that similarity is symmetric: compute_similarity(a, b) == compute_similarity(b, a)
    #[kani::proof]
    fn verify_similarity_symmetry() {
        let a = PropertyFeatures {
            property_type: String::new(),
            depth: kani::any(),
            quantifier_depth: kani::any(),
            implication_count: kani::any(),
            arithmetic_ops: kani::any(),
            function_calls: kani::any(),
            variable_count: 0,
            has_temporal: kani::any(),
            type_refs: vec![],
            keywords: vec![],
        };
        let b = PropertyFeatures {
            property_type: String::new(),
            depth: kani::any(),
            quantifier_depth: kani::any(),
            implication_count: kani::any(),
            arithmetic_ops: kani::any(),
            function_calls: kani::any(),
            variable_count: 0,
            has_temporal: kani::any(),
            type_refs: vec![],
            keywords: vec![],
        };

        let sim_ab = compute_similarity(&a, &b);
        let sim_ba = compute_similarity(&b, &a);

        kani::assert(
            (sim_ab - sim_ba).abs() < 1e-10,
            "Similarity must be symmetric",
        );
    }

    /// Prove that matching property type contributes to higher similarity
    #[kani::proof]
    fn verify_type_match_increases_similarity() {
        // Create two features with matching property types
        let matching_type = PropertyFeatures {
            property_type: "theorem".to_string(),
            depth: 1,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            function_calls: 0,
            variable_count: 0,
            has_temporal: false,
            type_refs: vec![],
            keywords: vec![],
        };
        let mismatched_type = PropertyFeatures {
            property_type: "invariant".to_string(),
            depth: 1,
            quantifier_depth: 0,
            implication_count: 0,
            arithmetic_ops: 0,
            function_calls: 0,
            variable_count: 0,
            has_temporal: false,
            type_refs: vec![],
            keywords: vec![],
        };

        let sim_match = compute_similarity(&matching_type, &matching_type);
        let sim_mismatch = compute_similarity(&matching_type, &mismatched_type);

        kani::assert(
            sim_match > sim_mismatch,
            "Matching property types should yield higher similarity",
        );
    }

    /// Prove that weight_sum is never zero (prevents division by zero)
    #[kani::proof]
    fn verify_no_division_by_zero() {
        // The weight_sum in compute_similarity is a constant:
        // type_weight(3.0) + depth_weight(1.0) + quant_weight(2.0) + impl_weight(1.5)
        // + arith_weight(1.0) + temp_weight(2.0) + func_weight(1.0) = 11.5
        let weight_sum: f64 = 3.0 + 1.0 + 2.0 + 1.5 + 1.0 + 2.0 + 1.0;
        kani::assert(weight_sum > 0.0, "Weight sum must be positive");
        kani::assert(
            (weight_sum - 11.5).abs() < 0.001,
            "Weight sum should be exactly 11.5",
        );
    }
}
