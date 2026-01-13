//! Feature extraction for ML-based strategy prediction
//!
//! This module provides feature extraction from USL properties for use
//! in the neural network-based strategy predictor.

use dashprove_usl::ast::{BinaryOp, Expr, Property, TemporalExpr};
use serde::{Deserialize, Serialize};

/// Number of input features for the neural network
pub const NUM_FEATURES: usize = 32;

/// Number of supported backends (output classes for backend prediction)
/// Updated to 113 for Phase 12 Additional tools (NuSMV, CPAchecker, SeaHorn, Frama-C)
pub const NUM_BACKENDS: usize = 113;

/// Maximum tactics to predict
pub const MAX_TACTICS: usize = 10;

/// Feature vector extracted from a property for ML prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyFeatureVector {
    /// Raw feature values (normalized to 0.0-1.0)
    pub features: Vec<f64>,
    /// Feature names for interpretability
    pub feature_names: Vec<String>,
}

impl PropertyFeatureVector {
    /// Extract features from a property for ML model input
    pub fn from_property(property: &Property) -> Self {
        let mut features = vec![0.0; NUM_FEATURES];
        let mut feature_names = vec![String::new(); NUM_FEATURES];

        // Feature indices
        const F_PROP_THEOREM: usize = 0;
        const F_PROP_TEMPORAL: usize = 1;
        const F_PROP_CONTRACT: usize = 2;
        const F_PROP_INVARIANT: usize = 3;
        const F_PROP_REFINEMENT: usize = 4;
        const F_PROP_PROBABILISTIC: usize = 5;
        const F_PROP_SECURITY: usize = 6;
        const F_QUANTIFIER_DEPTH: usize = 7;
        const F_FORALL_COUNT: usize = 8;
        const F_EXISTS_COUNT: usize = 9;
        const F_IMPLICATION_COUNT: usize = 10;
        const F_NEGATION_COUNT: usize = 11;
        const F_CONJUNCTION_COUNT: usize = 12;
        const F_DISJUNCTION_COUNT: usize = 13;
        const F_ARITHMETIC_OPS: usize = 14;
        const F_COMPARISON_OPS: usize = 15;
        const F_FUNCTION_CALLS: usize = 16;
        const F_METHOD_CALLS: usize = 17;
        const F_FIELD_ACCESS: usize = 18;
        const F_FORALL_IN_COUNT: usize = 19;
        const F_TEMPORAL_ALWAYS: usize = 20;
        const F_TEMPORAL_EVENTUALLY: usize = 21;
        const F_TEMPORAL_LEADS_TO: usize = 22;
        const F_RESERVED_1: usize = 23;
        const F_MAX_DEPTH: usize = 24;
        const F_NUM_VARIABLES: usize = 25;
        const F_NUM_CONSTANTS: usize = 26;
        const F_HAS_RECURSION: usize = 27;
        const F_HAS_INDUCTION: usize = 28;
        const F_COMPLEXITY_SCORE: usize = 29;
        const F_NAME_LENGTH: usize = 30;
        const F_HAS_PRECONDITION: usize = 31;

        // Set feature names
        feature_names[F_PROP_THEOREM] = "prop_theorem".to_string();
        feature_names[F_PROP_TEMPORAL] = "prop_temporal".to_string();
        feature_names[F_PROP_CONTRACT] = "prop_contract".to_string();
        feature_names[F_PROP_INVARIANT] = "prop_invariant".to_string();
        feature_names[F_PROP_REFINEMENT] = "prop_refinement".to_string();
        feature_names[F_PROP_PROBABILISTIC] = "prop_probabilistic".to_string();
        feature_names[F_PROP_SECURITY] = "prop_security".to_string();
        feature_names[F_QUANTIFIER_DEPTH] = "quantifier_depth".to_string();
        feature_names[F_FORALL_COUNT] = "forall_count".to_string();
        feature_names[F_EXISTS_COUNT] = "exists_count".to_string();
        feature_names[F_IMPLICATION_COUNT] = "implication_count".to_string();
        feature_names[F_NEGATION_COUNT] = "negation_count".to_string();
        feature_names[F_CONJUNCTION_COUNT] = "conjunction_count".to_string();
        feature_names[F_DISJUNCTION_COUNT] = "disjunction_count".to_string();
        feature_names[F_ARITHMETIC_OPS] = "arithmetic_ops".to_string();
        feature_names[F_COMPARISON_OPS] = "comparison_ops".to_string();
        feature_names[F_FUNCTION_CALLS] = "function_calls".to_string();
        feature_names[F_METHOD_CALLS] = "method_calls".to_string();
        feature_names[F_FIELD_ACCESS] = "field_access".to_string();
        feature_names[F_FORALL_IN_COUNT] = "forall_in_count".to_string();
        feature_names[F_TEMPORAL_ALWAYS] = "temporal_always".to_string();
        feature_names[F_TEMPORAL_EVENTUALLY] = "temporal_eventually".to_string();
        feature_names[F_TEMPORAL_LEADS_TO] = "temporal_leads_to".to_string();
        feature_names[F_RESERVED_1] = "reserved_1".to_string();
        feature_names[F_MAX_DEPTH] = "max_depth".to_string();
        feature_names[F_NUM_VARIABLES] = "num_variables".to_string();
        feature_names[F_NUM_CONSTANTS] = "num_constants".to_string();
        feature_names[F_HAS_RECURSION] = "has_recursion".to_string();
        feature_names[F_HAS_INDUCTION] = "has_induction".to_string();
        feature_names[F_COMPLEXITY_SCORE] = "complexity_score".to_string();
        feature_names[F_NAME_LENGTH] = "name_length".to_string();
        feature_names[F_HAS_PRECONDITION] = "has_precondition".to_string();

        // Extract features based on property type
        let (expr_features, name_len) = match property {
            Property::Theorem(t) => {
                features[F_PROP_THEOREM] = 1.0;
                (extract_expr_features(&t.body), t.name.len())
            }
            Property::Temporal(t) => {
                features[F_PROP_TEMPORAL] = 1.0;
                let mut ef = ExprFeatures::default();
                extract_temporal_features(&t.body, 0, &mut ef);
                (ef, t.name.len())
            }
            Property::Contract(c) => {
                features[F_PROP_CONTRACT] = 1.0;
                features[F_HAS_PRECONDITION] = if !c.requires.is_empty() { 1.0 } else { 0.0 };
                // Get contract name length from type_path
                let name_len = c.type_path.last().map(|s| s.len()).unwrap_or(8);
                // Combine ensures into feature extraction
                let mut ef = ExprFeatures::default();
                for expr in &c.ensures {
                    extract_expr_features_recursive(expr, 0, &mut ef);
                }
                for expr in &c.requires {
                    extract_expr_features_recursive(expr, 0, &mut ef);
                }
                (ef, name_len)
            }
            Property::Invariant(i) => {
                features[F_PROP_INVARIANT] = 1.0;
                (extract_expr_features(&i.body), i.name.len())
            }
            Property::Refinement(r) => {
                features[F_PROP_REFINEMENT] = 1.0;
                (extract_expr_features(&r.abstraction), r.name.len())
            }
            Property::Probabilistic(p) => {
                features[F_PROP_PROBABILISTIC] = 1.0;
                (extract_expr_features(&p.condition), p.name.len())
            }
            Property::Security(s) => {
                features[F_PROP_SECURITY] = 1.0;
                (extract_expr_features(&s.body), s.name.len())
            }
            Property::Semantic(s) => {
                features[F_PROP_THEOREM] = 1.0;
                (extract_expr_features(&s.body), s.name.len())
            }
            Property::PlatformApi(p) => {
                // PlatformApi is a meta-constraint; treat like invariant for feature extraction
                features[F_PROP_INVARIANT] = 1.0;
                (ExprFeatures::default(), p.name.len())
            }
            Property::Bisimulation(b) => {
                // Bisimulation is behavioral equivalence; treat like refinement for feature extraction
                features[F_PROP_REFINEMENT] = 1.0;
                (ExprFeatures::default(), b.name.len())
            }
            Property::Version(v) => {
                // Version improvement is like refinement for feature extraction
                features[F_PROP_REFINEMENT] = 1.0;
                let mut ef = ExprFeatures::default();
                for cap in &v.capabilities {
                    extract_expr_features_recursive(&cap.expr, 0, &mut ef);
                }
                for pres in &v.preserves {
                    extract_expr_features_recursive(&pres.property, 0, &mut ef);
                }
                (ef, v.name.len())
            }
            Property::Capability(c) => {
                // Capability specs are safety properties for feature extraction
                features[F_PROP_INVARIANT] = 1.0;
                let mut ef = ExprFeatures::default();
                for req in &c.requires {
                    extract_expr_features_recursive(req, 0, &mut ef);
                }
                (ef, c.name.len())
            }
            Property::DistributedInvariant(d) => {
                // Distributed invariants are like invariants for feature extraction
                features[F_PROP_INVARIANT] = 1.0;
                (extract_expr_features(&d.body), d.name.len())
            }
            Property::DistributedTemporal(d) => {
                // Distributed temporal are like temporal for feature extraction
                features[F_PROP_TEMPORAL] = 1.0;
                let mut ef = ExprFeatures::default();
                extract_temporal_features(&d.body, 0, &mut ef);
                (ef, d.name.len())
            }
            Property::Composed(c) => {
                // Composed theorems are like theorems for feature extraction
                features[F_PROP_THEOREM] = 1.0;
                (extract_expr_features(&c.body), c.name.len())
            }
            Property::ImprovementProposal(i) => {
                // Improvement proposals are like refinements for feature extraction
                features[F_PROP_REFINEMENT] = 1.0;
                let mut ef = ExprFeatures::default();
                for improves_expr in &i.improves {
                    extract_expr_features_recursive(improves_expr, 0, &mut ef);
                }
                for preserves_expr in &i.preserves {
                    extract_expr_features_recursive(preserves_expr, 0, &mut ef);
                }
                (ef, i.name.len())
            }
            Property::VerificationGate(v) => {
                // Verification gates are like invariants for feature extraction
                features[F_PROP_INVARIANT] = 1.0;
                let mut ef = ExprFeatures::default();
                for check in &v.checks {
                    extract_expr_features_recursive(&check.condition, 0, &mut ef);
                }
                (ef, v.name.len())
            }
            Property::Rollback(r) => {
                // Rollback specs are like temporal properties for feature extraction
                features[F_PROP_TEMPORAL] = 1.0;
                let mut ef = ExprFeatures::default();
                extract_expr_features_recursive(&r.trigger, 0, &mut ef);
                for guarantee in &r.guarantees {
                    extract_expr_features_recursive(guarantee, 0, &mut ef);
                }
                (ef, r.name.len())
            }
        };

        features[F_QUANTIFIER_DEPTH] = normalize(expr_features.quantifier_depth as f64, 10.0);
        features[F_FORALL_COUNT] = normalize(expr_features.forall_count as f64, 20.0);
        features[F_EXISTS_COUNT] = normalize(expr_features.exists_count as f64, 20.0);
        features[F_IMPLICATION_COUNT] = normalize(expr_features.implication_count as f64, 20.0);
        features[F_NEGATION_COUNT] = normalize(expr_features.negation_count as f64, 20.0);
        features[F_CONJUNCTION_COUNT] = normalize(expr_features.conjunction_count as f64, 50.0);
        features[F_DISJUNCTION_COUNT] = normalize(expr_features.disjunction_count as f64, 50.0);
        features[F_ARITHMETIC_OPS] = normalize(expr_features.arithmetic_ops as f64, 30.0);
        features[F_COMPARISON_OPS] = normalize(expr_features.comparison_ops as f64, 30.0);
        features[F_FUNCTION_CALLS] = normalize(expr_features.function_calls as f64, 50.0);
        features[F_METHOD_CALLS] = normalize(expr_features.method_calls as f64, 20.0);
        features[F_FIELD_ACCESS] = normalize(expr_features.field_access as f64, 20.0);
        features[F_FORALL_IN_COUNT] = normalize(expr_features.forall_in_count as f64, 20.0);
        features[F_TEMPORAL_ALWAYS] = normalize(expr_features.temporal_always as f64, 10.0);
        features[F_TEMPORAL_EVENTUALLY] = normalize(expr_features.temporal_eventually as f64, 10.0);
        features[F_TEMPORAL_LEADS_TO] = normalize(expr_features.temporal_leads_to as f64, 10.0);
        features[F_MAX_DEPTH] = normalize(expr_features.max_depth as f64, 20.0);
        features[F_NUM_VARIABLES] = normalize(expr_features.num_variables as f64, 50.0);
        features[F_NUM_CONSTANTS] = normalize(expr_features.num_constants as f64, 30.0);
        features[F_HAS_RECURSION] = if expr_features.has_recursion {
            1.0
        } else {
            0.0
        };
        features[F_HAS_INDUCTION] = if expr_features.has_induction {
            1.0
        } else {
            0.0
        };
        features[F_COMPLEXITY_SCORE] = normalize(expr_features.complexity_score(), 100.0);
        features[F_NAME_LENGTH] = normalize(name_len as f64, 100.0);

        PropertyFeatureVector {
            features,
            feature_names,
        }
    }

    /// Get feature value by name
    pub fn get(&self, name: &str) -> Option<f64> {
        self.feature_names
            .iter()
            .position(|n| n == name)
            .map(|i| self.features[i])
    }
}

/// Raw expression features before normalization
#[derive(Debug, Default)]
struct ExprFeatures {
    quantifier_depth: usize,
    forall_count: usize,
    exists_count: usize,
    forall_in_count: usize,
    implication_count: usize,
    negation_count: usize,
    conjunction_count: usize,
    disjunction_count: usize,
    arithmetic_ops: usize,
    comparison_ops: usize,
    function_calls: usize,
    method_calls: usize,
    field_access: usize,
    temporal_always: usize,
    temporal_eventually: usize,
    temporal_leads_to: usize,
    max_depth: usize,
    num_variables: usize,
    num_constants: usize,
    has_recursion: bool,
    has_induction: bool,
}

impl ExprFeatures {
    fn complexity_score(&self) -> f64 {
        (self.quantifier_depth as f64 * 3.0)
            + (self.forall_count as f64 * 2.0)
            + (self.exists_count as f64 * 2.5)
            + (self.implication_count as f64 * 1.5)
            + (self.function_calls as f64 * 1.0)
            + (self.arithmetic_ops as f64 * 0.5)
            + (self.max_depth as f64 * 1.0)
            + (self.temporal_always as f64 * 2.0)
            + (self.temporal_eventually as f64 * 2.0)
            + if self.has_recursion { 10.0 } else { 0.0 }
            + if self.has_induction { 8.0 } else { 0.0 }
    }
}

/// Extract features from an expression
fn extract_expr_features(expr: &Expr) -> ExprFeatures {
    let mut features = ExprFeatures::default();
    extract_expr_features_recursive(expr, 0, &mut features);
    features
}

fn extract_expr_features_recursive(expr: &Expr, depth: usize, features: &mut ExprFeatures) {
    features.max_depth = features.max_depth.max(depth);

    match expr {
        Expr::ForAll { body, .. } => {
            features.forall_count += 1;
            features.quantifier_depth = features.quantifier_depth.max(depth + 1);
            extract_expr_features_recursive(body, depth + 1, features);
        }
        Expr::Exists { body, .. } => {
            features.exists_count += 1;
            features.quantifier_depth = features.quantifier_depth.max(depth + 1);
            extract_expr_features_recursive(body, depth + 1, features);
        }
        Expr::ForAllIn {
            body, collection, ..
        } => {
            features.forall_count += 1;
            features.forall_in_count += 1;
            features.quantifier_depth = features.quantifier_depth.max(depth + 1);
            extract_expr_features_recursive(collection, depth + 1, features);
            extract_expr_features_recursive(body, depth + 1, features);
        }
        Expr::ExistsIn {
            body, collection, ..
        } => {
            features.exists_count += 1;
            features.forall_in_count += 1;
            features.quantifier_depth = features.quantifier_depth.max(depth + 1);
            extract_expr_features_recursive(collection, depth + 1, features);
            extract_expr_features_recursive(body, depth + 1, features);
        }
        Expr::Implies(lhs, rhs) => {
            features.implication_count += 1;
            extract_expr_features_recursive(lhs, depth + 1, features);
            extract_expr_features_recursive(rhs, depth + 1, features);
        }
        Expr::Not(inner) => {
            features.negation_count += 1;
            extract_expr_features_recursive(inner, depth + 1, features);
        }
        Expr::And(lhs, rhs) => {
            features.conjunction_count += 1;
            extract_expr_features_recursive(lhs, depth + 1, features);
            extract_expr_features_recursive(rhs, depth + 1, features);
        }
        Expr::Or(lhs, rhs) => {
            features.disjunction_count += 1;
            extract_expr_features_recursive(lhs, depth + 1, features);
            extract_expr_features_recursive(rhs, depth + 1, features);
        }
        Expr::Binary(lhs, op, rhs) => {
            match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                    features.arithmetic_ops += 1;
                }
            }
            extract_expr_features_recursive(lhs, depth + 1, features);
            extract_expr_features_recursive(rhs, depth + 1, features);
        }
        Expr::Compare(lhs, _op, rhs) => {
            features.comparison_ops += 1;
            extract_expr_features_recursive(lhs, depth + 1, features);
            extract_expr_features_recursive(rhs, depth + 1, features);
        }
        Expr::App(_, args) => {
            features.function_calls += 1;
            for arg in args {
                extract_expr_features_recursive(arg, depth + 1, features);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            features.method_calls += 1;
            extract_expr_features_recursive(receiver, depth + 1, features);
            for arg in args {
                extract_expr_features_recursive(arg, depth + 1, features);
            }
        }
        Expr::FieldAccess(inner, _) => {
            features.field_access += 1;
            extract_expr_features_recursive(inner, depth + 1, features);
        }
        Expr::Neg(inner) => {
            features.arithmetic_ops += 1;
            extract_expr_features_recursive(inner, depth + 1, features);
        }
        Expr::Var(_) => {
            features.num_variables += 1;
        }
        Expr::Int(_) | Expr::Bool(_) | Expr::String(_) | Expr::Float(_) => {
            features.num_constants += 1;
        }
    }
}

/// Extract features from temporal expressions
fn extract_temporal_features(expr: &TemporalExpr, depth: usize, features: &mut ExprFeatures) {
    features.max_depth = features.max_depth.max(depth);

    match expr {
        TemporalExpr::Always(inner) => {
            features.temporal_always += 1;
            extract_temporal_features(inner, depth + 1, features);
        }
        TemporalExpr::Eventually(inner) => {
            features.temporal_eventually += 1;
            extract_temporal_features(inner, depth + 1, features);
        }
        TemporalExpr::LeadsTo(lhs, rhs) => {
            features.temporal_leads_to += 1;
            extract_temporal_features(lhs, depth + 1, features);
            extract_temporal_features(rhs, depth + 1, features);
        }
        TemporalExpr::Atom(expr) => {
            extract_expr_features_recursive(expr, depth + 1, features);
        }
    }
}

/// Normalize a value to 0.0-1.0 range using tanh-based scaling
pub fn normalize(value: f64, scale: f64) -> f64 {
    (value / scale).tanh()
}
