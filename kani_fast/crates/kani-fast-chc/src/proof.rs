//! Proof generation from CHC solving results
//!
//! This module converts CHC verification results into the universal proof format
//! for cross-backend proof sharing.

use crate::result::{ChcResult, InvariantModel, SolvedPredicate};
use crate::ChcSystem;
use kani_fast_proof::{BackendId, ChcStep, ProofFormat, ProofStep, UniversalProof};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

/// Cached Z3 version string
static Z3_VERSION: OnceLock<String> = OnceLock::new();

/// Detect Z3 version by running `z3 --version`
///
/// Caches the result for subsequent calls.
fn detect_z3_version() -> &'static str {
    Z3_VERSION.get_or_init(|| {
        Command::new("z3")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| {
                let stdout = String::from_utf8_lossy(&output.stdout);
                // Parse "Z3 version 4.15.4 - 64 bit" -> "4.15.4"
                stdout.split_whitespace().nth(2).map(|s| s.to_string())
            })
            .unwrap_or_else(|| "unknown".to_string())
    })
}

/// Generate a universal proof from a CHC verification result
///
/// # Arguments
/// * `result` - The CHC solving result
/// * `system` - The CHC system that was verified
/// * `generation_time` - Time spent generating this proof
///
/// # Returns
/// A `UniversalProof` containing CHC proof steps
pub fn generate_chc_proof(
    result: &ChcResult,
    system: &ChcSystem,
    generation_time: Duration,
) -> Option<UniversalProof> {
    match result {
        ChcResult::Sat { model, stats, .. } => {
            let proof = build_sat_proof(model, system, stats.solve_time, generation_time);
            Some(proof)
        }
        ChcResult::Unsat { .. } => {
            // For UNSAT (property violated), we don't have a proof of correctness
            // We could generate a proof of the violation, but that's a different use case
            None
        }
        ChcResult::Unknown { .. } => None,
    }
}

/// Build a SAT proof from the invariant model
fn build_sat_proof(
    model: &InvariantModel,
    system: &ChcSystem,
    solve_time: Duration,
    generation_time: Duration,
) -> UniversalProof {
    let vc = system.to_smt2();

    let mut builder = UniversalProof::builder()
        .backend(BackendId::Z3)
        .format(ProofFormat::Chc)
        .vc(&vc)
        .backend_version(detect_z3_version())
        .generation_time(generation_time);

    // Add invariant steps from the model
    for pred in &model.predicates {
        let step = predicate_to_step(pred);
        builder = builder.step(step);
    }

    // Add initiation step (Init => Inv)
    // Extract init condition from system (clauses with no body predicates)
    let init_conditions: Vec<String> = system
        .clauses
        .iter()
        .filter(|c| c.body_preds.is_empty())
        .map(|c| c.constraint.smt_formula.clone())
        .collect();

    if !init_conditions.is_empty() && !model.predicates.is_empty() {
        let init = init_conditions.join(" ∧ ");
        let inv = model.predicates[0].formula.smt_formula.clone();
        builder = builder.step(ProofStep::Chc(ChcStep::initiation(&init, &inv)));
    }

    // Add consecution step (Inv ∧ Trans => Inv')
    // Extract transition clauses (clauses with body predicates and non-query head)
    let trans_clauses: Vec<&crate::clause::HornClause> = system
        .clauses
        .iter()
        .filter(|c| !c.body_preds.is_empty() && !matches!(c.head, crate::clause::ClauseHead::Query))
        .collect();

    if !trans_clauses.is_empty() && !model.predicates.is_empty() {
        let pre_inv = model.predicates[0].formula.smt_formula.clone();
        let post_inv = format!("{}' (prime state)", pre_inv);
        let trans = format!("{} transitions", trans_clauses.len());
        builder = builder.step(ProofStep::Chc(ChcStep::consecution(
            &pre_inv, &trans, &post_inv,
        )));
    }

    // Add property step (Inv => Property)
    // Extract property clauses (head is Query/false)
    let prop_clauses: Vec<&crate::clause::HornClause> = system
        .clauses
        .iter()
        .filter(|c| matches!(c.head, crate::clause::ClauseHead::Query))
        .collect();

    if !prop_clauses.is_empty() && !model.predicates.is_empty() {
        let inv = model.predicates[0].formula.smt_formula.clone();
        // The property is the negation of the constraint that leads to false
        let property = prop_clauses
            .first()
            .map(|c| format!("not({})", c.constraint.smt_formula))
            .unwrap_or_else(|| "true".to_string());
        builder = builder.step(ProofStep::Chc(ChcStep::property(&inv, &property)));
    }

    // Add property name and description from solve time
    builder = builder
        .property_name("chc_verification")
        .description(format!("CHC verification completed in {:?}", solve_time));

    builder.build()
}

/// Convert a solved predicate to a CHC proof step
fn predicate_to_step(pred: &SolvedPredicate) -> ProofStep {
    let params: Vec<String> = pred.params.iter().map(|(n, _)| n).cloned().collect();
    ProofStep::chc_invariant(&pred.name, params, &pred.formula.smt_formula)
}

/// Generate proof metadata as JSON
pub fn proof_to_json(proof: &UniversalProof) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(proof)
}

/// A builder for CHC proofs with more control over proof structure
pub struct ChcProofBuilder {
    vc: String,
    steps: Vec<ProofStep>,
    property_name: Option<String>,
    description: Option<String>,
    backend: BackendId,
    generation_time: Duration,
}

impl Default for ChcProofBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ChcProofBuilder {
    /// Create a new CHC proof builder
    pub fn new() -> Self {
        Self {
            vc: String::new(),
            steps: Vec::new(),
            property_name: None,
            description: None,
            backend: BackendId::Z3,
            generation_time: Duration::ZERO,
        }
    }

    /// Set the verification condition
    pub fn vc(mut self, vc: impl Into<String>) -> Self {
        self.vc = vc.into();
        self
    }

    /// Add an invariant
    pub fn invariant(
        mut self,
        name: impl Into<String>,
        params: Vec<String>,
        formula: impl Into<String>,
    ) -> Self {
        self.steps
            .push(ProofStep::chc_invariant(name, params, formula));
        self
    }

    /// Add an initiation step
    pub fn initiation(mut self, init: impl Into<String>, inv: impl Into<String>) -> Self {
        self.steps
            .push(ProofStep::Chc(ChcStep::initiation(init, inv)));
        self
    }

    /// Add a consecution step
    pub fn consecution(
        mut self,
        pre_inv: impl Into<String>,
        trans: impl Into<String>,
        post_inv: impl Into<String>,
    ) -> Self {
        self.steps.push(ProofStep::Chc(ChcStep::consecution(
            pre_inv, trans, post_inv,
        )));
        self
    }

    /// Add a property step
    pub fn property(mut self, inv: impl Into<String>, property: impl Into<String>) -> Self {
        self.steps
            .push(ProofStep::Chc(ChcStep::property(inv, property)));
        self
    }

    /// Set the property name
    pub fn property_name(mut self, name: impl Into<String>) -> Self {
        self.property_name = Some(name.into());
        self
    }

    /// Set the description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the backend
    pub fn backend(mut self, backend: BackendId) -> Self {
        self.backend = backend;
        self
    }

    /// Set the generation time
    pub fn generation_time(mut self, time: Duration) -> Self {
        self.generation_time = time;
        self
    }

    /// Build the proof
    pub fn build(self) -> UniversalProof {
        let mut builder = UniversalProof::builder()
            .backend(self.backend)
            .format(ProofFormat::Chc)
            .vc(&self.vc)
            .steps(self.steps)
            .generation_time(self.generation_time);

        if let Some(name) = self.property_name {
            builder = builder.property_name(name);
        }
        if let Some(desc) = self.description {
            builder = builder.description(desc);
        }

        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clause::{ChcSystemBuilder, Variable};
    use crate::encoding::encode_simple_loop;
    use kani_fast_kinduction::SmtType;

    #[test]
    fn test_chc_proof_builder() {
        let proof = ChcProofBuilder::new()
            .vc("(assert (>= x 0))")
            .invariant("Inv", vec!["x".to_string()], "(>= x 0)")
            .initiation("(= x 0)", "(>= x 0)")
            .consecution("(>= x 0)", "(= x' (+ x 1))", "(>= x' 0)")
            .property("(>= x 0)", "(not (< x 0))")
            .property_name("counter_safety")
            .description("Proves counter is non-negative")
            .backend(BackendId::Z3)
            .generation_time(Duration::from_millis(50))
            .build();

        assert_eq!(proof.backend, BackendId::Z3);
        assert_eq!(proof.format, ProofFormat::Chc);
        assert_eq!(proof.steps.len(), 4);
        assert!(proof.verify_integrity());
        assert_eq!(
            proof.metadata.property_name,
            Some("counter_safety".to_string())
        );
    }

    #[test]
    fn test_generate_chc_proof_from_result() {
        use crate::result::{ChcSolverStats, SolvedPredicate};
        use kani_fast_kinduction::StateFormula;

        // Create a mock CHC result
        let model = InvariantModel {
            predicates: vec![SolvedPredicate {
                name: "Inv".to_string(),
                params: vec![("x".to_string(), SmtType::Int)],
                formula: StateFormula::new("(>= x 0)"),
            }],
        };

        let result = ChcResult::Sat {
            model,
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        let system = encode_simple_loop("x", SmtType::Int, "(= x 0)", "(= x' (+ x 1))", "(>= x 0)");

        let proof = generate_chc_proof(&result, &system, Duration::from_millis(100));

        assert!(proof.is_some());
        let proof = proof.unwrap();
        assert_eq!(proof.backend, BackendId::Z3);
        assert_eq!(proof.format, ProofFormat::Chc);
        assert!(!proof.steps.is_empty());
        assert!(proof.verify_integrity());
    }

    #[test]
    fn test_generate_chc_proof_unsat_returns_none() {
        use crate::result::ChcSolverStats;

        let result = ChcResult::Unsat {
            counterexample: None,
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        let system = encode_simple_loop("x", SmtType::Int, "(= x 5)", "(= x' (- x 1))", "(>= x 0)");

        let proof = generate_chc_proof(&result, &system, Duration::from_millis(50));

        assert!(proof.is_none());
    }

    #[test]
    fn test_generate_chc_proof_counts_multiple_transitions() {
        use crate::result::{ChcSolverStats, SolvedPredicate};
        use kani_fast_kinduction::StateFormula;

        let model = InvariantModel {
            predicates: vec![SolvedPredicate {
                name: "Inv".to_string(),
                params: vec![("x".to_string(), SmtType::Int)],
                formula: StateFormula::new("(>= x 0)"),
            }],
        };

        let result = ChcResult::Sat {
            model,
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        let current_vars = vec![Variable::new("x", SmtType::Int)];
        let transition_vars = vec![
            Variable::new("x", SmtType::Int),
            Variable::new("x_next", SmtType::Int),
        ];

        let system = ChcSystemBuilder::new()
            .predicate("Inv", vec![SmtType::Int])
            .init(
                "Inv",
                vec!["x".to_string()],
                current_vars.clone(),
                "(= x 0)",
            )
            .transition(
                "Inv",
                vec!["x".to_string()],
                vec!["x_next".to_string()],
                transition_vars.clone(),
                "(= x_next (+ x 1))",
            )
            .transition(
                "Inv",
                vec!["x".to_string()],
                vec!["x_next".to_string()],
                transition_vars,
                "(= x_next (+ x 2))",
            )
            .property("Inv", vec!["x".to_string()], current_vars, "(>= x 0)")
            .build();

        let proof = generate_chc_proof(&result, &system, Duration::from_millis(25)).unwrap();

        let consecution_transition = proof.steps.iter().find_map(|step| match step {
            ProofStep::Chc(ChcStep::Consecution { transition, .. }) => Some(transition),
            _ => None,
        });

        assert_eq!(consecution_transition, Some(&"2 transitions".to_string()));
    }

    #[test]
    fn test_generate_chc_proof_unknown_returns_none() {
        use crate::result::ChcSolverStats;

        let result = ChcResult::Unknown {
            reason: "timeout".to_string(),
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        let system = encode_simple_loop("x", SmtType::Int, "(= x 0)", "(= x' (+ x 1))", "(>= x 0)");

        let proof = generate_chc_proof(&result, &system, Duration::from_millis(50));

        assert!(proof.is_none());
    }

    #[test]
    fn test_property_step_uses_query_constraint() {
        use crate::result::{ChcSolverStats, SolvedPredicate};
        use kani_fast_kinduction::StateFormula;

        let model = InvariantModel {
            predicates: vec![SolvedPredicate {
                name: "Inv".to_string(),
                params: vec![("x".to_string(), SmtType::Int)],
                formula: StateFormula::new("(>= x 0)"),
            }],
        };

        let result = ChcResult::Sat {
            model,
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        let system = encode_simple_loop("x", SmtType::Int, "(= x 0)", "(= x' (+ x 1))", "(>= x 0)");
        let proof = generate_chc_proof(&result, &system, Duration::from_millis(10)).unwrap();

        let property = proof.steps.iter().find_map(|step| match step {
            ProofStep::Chc(ChcStep::Property { property, .. }) => Some(property),
            _ => None,
        });

        assert_eq!(property, Some(&"not((not (>= x 0)))".to_string()));
    }

    #[test]
    fn test_proof_to_json() {
        let proof = ChcProofBuilder::new()
            .vc("(assert true)")
            .invariant("Inv", vec![], "true")
            .build();

        let json = proof_to_json(&proof);
        assert!(json.is_ok());

        let json_str = json.unwrap();
        assert!(json_str.contains("\"backend\""));
        assert!(json_str.contains("\"format\""));
        assert!(json_str.contains("\"steps\""));
    }

    #[test]
    fn test_predicate_to_step() {
        use kani_fast_kinduction::StateFormula;

        let pred = SolvedPredicate {
            name: "Inv".to_string(),
            params: vec![
                ("x".to_string(), SmtType::Int),
                ("y".to_string(), SmtType::Bool),
            ],
            formula: StateFormula::new("(and (>= x 0) y)"),
        };

        let step = predicate_to_step(&pred);

        if let ProofStep::Chc(ChcStep::Invariant {
            name,
            params,
            formula,
        }) = step
        {
            assert_eq!(name, "Inv");
            assert_eq!(params, vec!["x", "y"]);
            assert_eq!(formula, "(and (>= x 0) y)");
        } else {
            panic!("Expected CHC Invariant step");
        }
    }

    #[test]
    fn test_chc_proof_builder_default() {
        let builder = ChcProofBuilder::default();
        let proof = builder.build();

        assert_eq!(proof.backend, BackendId::Z3);
        assert_eq!(proof.format, ProofFormat::Chc);
        assert!(proof.steps.is_empty());
    }

    #[test]
    fn test_generate_chc_proof_with_empty_model_produces_metadata() {
        use crate::result::ChcSolverStats;

        let result = ChcResult::Sat {
            model: InvariantModel::default(),
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        let system = encode_simple_loop("x", SmtType::Int, "(= x 0)", "(= x' x)", "(>= x 0)");
        let proof = generate_chc_proof(&result, &system, Duration::from_millis(5)).unwrap();

        assert!(proof.steps.is_empty());
        assert_eq!(
            proof.metadata.property_name,
            Some("chc_verification".to_string())
        );
        assert!(proof
            .metadata
            .description
            .as_deref()
            .is_some_and(|desc| desc.contains("completed")));
    }

    #[test]
    fn test_proof_content_addressable() {
        let proof1 = ChcProofBuilder::new()
            .vc("(assert P)")
            .invariant("Inv", vec![], "(>= x 0)")
            .build();

        let proof2 = ChcProofBuilder::new()
            .vc("(assert P)")
            .invariant("Inv", vec![], "(>= x 0)")
            .build();

        // Same content should produce same ID
        assert_eq!(proof1.id, proof2.id);

        let proof3 = ChcProofBuilder::new()
            .vc("(assert Q)")
            .invariant("Inv", vec![], "(>= x 0)")
            .build();

        // Different VC should produce different ID
        assert_ne!(proof1.id, proof3.id);
    }

    #[test]
    fn test_detect_z3_version() {
        let version = detect_z3_version();
        // Version should be either a valid version string or "unknown"
        assert!(
            version == "unknown" || version.contains('.'),
            "Expected version string like 'x.y.z' or 'unknown', got: {}",
            version
        );
    }

    #[test]
    fn test_proof_uses_detected_version() {
        use crate::result::{ChcSolverStats, SolvedPredicate};
        use kani_fast_kinduction::StateFormula;

        let model = InvariantModel {
            predicates: vec![SolvedPredicate {
                name: "Inv".to_string(),
                params: vec![("x".to_string(), SmtType::Int)],
                formula: StateFormula::new("(>= x 0)"),
            }],
        };

        let result = ChcResult::Sat {
            model,
            stats: ChcSolverStats::default(),
            raw_output: None,
        };

        let system = encode_simple_loop("x", SmtType::Int, "(= x 0)", "(= x' (+ x 1))", "(>= x 0)");
        let proof = generate_chc_proof(&result, &system, Duration::from_millis(100)).unwrap();

        // Proof should use detected Z3 version
        let detected = detect_z3_version();
        assert_eq!(proof.metadata.backend_version, detected);
    }
}
