//! Proof generation from k-induction verification results
//!
//! This module converts k-induction verification results into the universal proof format
//! for cross-backend proof sharing.

use crate::formula::TransitionSystem;
use crate::result::{KInductionResult, KInductionStats};
use kani_fast_proof::{BackendId, ChcStep, ProofFormat, ProofStep, SmtStep, UniversalProof};
use std::fmt::Write;
use std::time::Duration;

/// Generate a universal proof from a k-induction verification result
///
/// # Arguments
/// * `result` - The k-induction result
/// * `system` - The transition system that was verified
/// * `generation_time` - Time spent generating this proof
///
/// # Returns
/// A `UniversalProof` containing the k-induction proof steps
pub fn generate_kinduction_proof(
    result: &KInductionResult,
    system: &TransitionSystem,
    generation_time: Duration,
) -> Option<UniversalProof> {
    match result {
        KInductionResult::Proven {
            k,
            invariant,
            stats,
        } => {
            let proof =
                build_proven_proof(*k, invariant.as_deref(), system, stats, generation_time);
            Some(proof)
        }
        KInductionResult::Disproven { .. } => {
            // Disproven results don't have proofs of correctness
            None
        }
        KInductionResult::Unknown { .. } => None,
    }
}

/// Build a proof for a proven property
fn build_proven_proof(
    k: u32,
    invariant: Option<&str>,
    system: &TransitionSystem,
    stats: &KInductionStats,
    generation_time: Duration,
) -> UniversalProof {
    // Generate VC from transition system
    let vc = generate_vc_from_system(system);

    let mut builder = UniversalProof::builder()
        .backend(BackendId::KaniFast)
        .format(ProofFormat::Mixed) // k-induction uses both SMT and CHC-like steps
        .vc(&vc)
        .backend_version("0.1.0")
        .generation_time(generation_time);

    // Add base case steps
    for i in 0..k {
        let step = ProofStep::Smt(SmtStep::assume(
            format!("base_case_{i}"),
            format!("property holds at step {i}"),
        ));
        builder = builder.step(step);
    }

    // Add induction hypothesis
    let induction_hyp = ProofStep::Smt(SmtStep::assume(
        "induction_hypothesis",
        format!("property holds for {k} consecutive states"),
    ));
    builder = builder.step(induction_hyp);

    // Add induction step
    let induction_step = ProofStep::Smt(SmtStep::infer(
        "induction_step",
        (0..k as usize).collect(),
        format!("property holds at state k+1 (k={k})"),
    ));
    builder = builder.step(induction_step);

    // If we have an invariant, add it as a CHC step
    if let Some(inv) = invariant {
        let inv_step = ProofStep::Chc(ChcStep::invariant(
            "discovered_invariant",
            vec!["state".to_string()],
            inv,
        ));
        builder = builder.step(inv_step);

        // Add initiation proof
        builder = builder.step(ProofStep::Chc(ChcStep::initiation(
            &system.init.smt_formula,
            inv,
        )));

        // Add consecution proof
        builder = builder.step(ProofStep::Chc(ChcStep::consecution(
            inv,
            &system.transition.smt_formula,
            inv,
        )));
    }

    // Add property verification
    for prop in &system.properties {
        let prop_step = ProofStep::Chc(ChcStep::property(
            invariant.unwrap_or("induction"),
            &prop.formula.smt_formula,
        ));
        builder = builder.step(prop_step);
    }

    // Set metadata
    let property_names: Vec<String> = system.properties.iter().map(|p| p.name.clone()).collect();
    builder = builder
        .property_name(property_names.join(", "))
        .description(format!(
            "K-induction proof (k={}) completed in {:?}",
            k, stats.total_time
        ));

    builder.build()
}

/// Generate verification condition from transition system
fn generate_vc_from_system(system: &TransitionSystem) -> String {
    let mut vc = String::new();

    // Declare variables
    for var in &system.variables {
        let _ = writeln!(
            vc,
            "(declare-const {} {})",
            var.name,
            var.smt_type.to_smt_string()
        );
    }

    // Assert init
    let _ = writeln!(vc, "(assert {})", system.init.smt_formula);

    // Assert transition
    let _ = writeln!(vc, "(assert {})", system.transition.smt_formula);

    // Assert properties
    for prop in &system.properties {
        let _ = writeln!(vc, "(assert {})", prop.formula.smt_formula);
    }

    vc
}

/// A builder for k-induction proofs with more control over proof structure
pub struct KInductionProofBuilder {
    vc: String,
    steps: Vec<ProofStep>,
    property_name: Option<String>,
    description: Option<String>,
    k: u32,
    invariant: Option<String>,
    generation_time: Duration,
}

impl Default for KInductionProofBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl KInductionProofBuilder {
    /// Create a new k-induction proof builder
    pub fn new() -> Self {
        Self {
            vc: String::new(),
            steps: Vec::new(),
            property_name: None,
            description: None,
            k: 1,
            invariant: None,
            generation_time: Duration::ZERO,
        }
    }

    /// Set the verification condition
    pub fn vc(mut self, vc: impl Into<String>) -> Self {
        self.vc = vc.into();
        self
    }

    /// Set the k value
    pub fn k(mut self, k: u32) -> Self {
        self.k = k;
        self
    }

    /// Add an invariant
    pub fn invariant(mut self, inv: impl Into<String>) -> Self {
        self.invariant = Some(inv.into());
        self
    }

    /// Add a base case step
    pub fn base_case(mut self, step: u32, formula: impl Into<String>) -> Self {
        self.steps.push(ProofStep::Smt(SmtStep::assume(
            format!("base_case_{step}"),
            formula.into(),
        )));
        self
    }

    /// Add the induction hypothesis
    pub fn induction_hypothesis(mut self, formula: impl Into<String>) -> Self {
        self.steps.push(ProofStep::Smt(SmtStep::assume(
            "induction_hypothesis",
            formula.into(),
        )));
        self
    }

    /// Add the induction step
    pub fn induction_step(
        mut self,
        premise_indices: Vec<usize>,
        conclusion: impl Into<String>,
    ) -> Self {
        self.steps.push(ProofStep::Smt(SmtStep::infer(
            "induction_step",
            premise_indices,
            conclusion.into(),
        )));
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

    /// Set the generation time
    pub fn generation_time(mut self, time: Duration) -> Self {
        self.generation_time = time;
        self
    }

    /// Build the proof
    pub fn build(self) -> UniversalProof {
        let mut builder = UniversalProof::builder()
            .backend(BackendId::KaniFast)
            .format(ProofFormat::Mixed)
            .vc(&self.vc)
            .steps(self.steps)
            .generation_time(self.generation_time);

        if let Some(name) = self.property_name {
            builder = builder.property_name(name);
        }
        if let Some(desc) = self.description {
            builder = builder.description(desc);
        }

        // Add invariant step if present
        if let Some(inv) = &self.invariant {
            builder = builder.step(ProofStep::chc_invariant(
                "discovered_invariant",
                vec!["state".to_string()],
                inv,
            ));
        }

        builder.build()
    }
}

/// Generate proof metadata as JSON
pub fn proof_to_json(proof: &UniversalProof) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(proof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::{SmtType, TransitionSystemBuilder};

    #[test]
    fn test_kinduction_proof_builder() {
        let proof = KInductionProofBuilder::new()
            .vc("(assert (>= x 0))")
            .k(3)
            .base_case(0, "(>= x_0 0)")
            .base_case(1, "(>= x_1 0)")
            .base_case(2, "(>= x_2 0)")
            .induction_hypothesis("(forall k. (>= x_k 0))")
            .induction_step(vec![0, 1, 2], "(>= x_3 0)")
            .invariant("(>= x 0)")
            .property_name("nonnegative")
            .description("Counter is always non-negative")
            .generation_time(Duration::from_millis(50))
            .build();

        assert_eq!(proof.backend, BackendId::KaniFast);
        assert_eq!(proof.format, ProofFormat::Mixed);
        assert!(!proof.steps.is_empty());
        assert!(proof.verify_integrity());
        assert_eq!(
            proof.metadata.property_name,
            Some("nonnegative".to_string())
        );
    }

    #[test]
    fn test_generate_kinduction_proof_proven() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "nonneg", "(>= x 0)")
            .build();

        let result = KInductionResult::proven_with_invariant(
            5,
            "(>= x 0)".to_string(),
            KInductionStats::default(),
        );

        let proof = generate_kinduction_proof(&result, &ts, Duration::from_millis(100));

        assert!(proof.is_some());
        let proof = proof.unwrap();
        assert_eq!(proof.backend, BackendId::KaniFast);
        assert!(!proof.steps.is_empty());
        assert!(proof.verify_integrity());
    }

    #[test]
    fn test_generate_kinduction_proof_disproven_returns_none() {
        use crate::result::Counterexample;

        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 5)")
            .transition("(= x' (- x 1))")
            .property("p1", "nonneg", "(>= x 0)")
            .build();

        let cex = Counterexample::new(vec![], 6, "(>= x 0)");
        let result = KInductionResult::disproven(6, cex, KInductionStats::default());

        let proof = generate_kinduction_proof(&result, &ts, Duration::from_millis(50));

        assert!(proof.is_none());
    }

    #[test]
    fn test_generate_kinduction_proof_unknown_returns_none() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "nonneg", "(>= x 0)")
            .build();

        let result = KInductionResult::unknown("timeout", 10, KInductionStats::default());

        let proof = generate_kinduction_proof(&result, &ts, Duration::from_millis(50));

        assert!(proof.is_none());
    }

    #[test]
    fn test_proof_to_json() {
        let proof = KInductionProofBuilder::new()
            .vc("(assert true)")
            .k(1)
            .base_case(0, "true")
            .build();

        let json = proof_to_json(&proof);
        assert!(json.is_ok());

        let json_str = json.unwrap();
        assert!(json_str.contains("\"backend\""));
        assert!(json_str.contains("\"KaniFast\""));
    }

    #[test]
    fn test_kinduction_proof_builder_default() {
        let builder = KInductionProofBuilder::default();
        let proof = builder.build();

        assert_eq!(proof.backend, BackendId::KaniFast);
        assert_eq!(proof.format, ProofFormat::Mixed);
    }

    #[test]
    fn test_proof_content_addressable() {
        let proof1 = KInductionProofBuilder::new()
            .vc("(assert P)")
            .k(3)
            .base_case(0, "P(0)")
            .build();

        let proof2 = KInductionProofBuilder::new()
            .vc("(assert P)")
            .k(3)
            .base_case(0, "P(0)")
            .build();

        // Same content should produce same ID
        assert_eq!(proof1.id, proof2.id);

        let proof3 = KInductionProofBuilder::new()
            .vc("(assert Q)")
            .k(3)
            .base_case(0, "Q(0)")
            .build();

        // Different VC should produce different ID
        assert_ne!(proof1.id, proof3.id);
    }

    #[test]
    fn test_generate_vc_from_system() {
        let ts = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Bool)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .property("p1", "nonneg", "(>= x 0)")
            .build();

        let vc = generate_vc_from_system(&ts);

        assert!(vc.contains("(declare-const x Int)"));
        assert!(vc.contains("(declare-const y Bool)"));
        assert!(vc.contains("(assert (= x 0))"));
        assert!(vc.contains("(assert (= x' (+ x 1)))"));
        assert!(vc.contains("(assert (>= x 0))"));
    }

    // ======== Mutation coverage tests ========

    #[test]
    fn test_builder_vc_method_returns_self() {
        // Line 189: vc() returns Self, not Default::default()
        let builder = KInductionProofBuilder::new().vc("(assert (>= x 0))");

        // Chain another method call to verify builder is returned
        let builder = builder.k(5);
        let proof = builder.build();

        // If vc() returned Default::default(), the VC would be empty
        assert!(proof.vc.contains("(assert (>= x 0))"));
    }

    #[test]
    fn test_builder_k_method_returns_self() {
        // Line 195: k() returns Self, not Default::default()
        let builder = KInductionProofBuilder::new().k(7);

        // Chain another method to verify k was set
        let builder = builder.vc("test");
        let proof = builder.build();

        // We can't directly check k on the proof, but we verify builder chaining works
        assert!(!proof.vc.is_empty());
    }

    #[test]
    fn test_builder_induction_hypothesis_returns_self() {
        // Line 216: induction_hypothesis() returns Self, not Default::default()
        let builder =
            KInductionProofBuilder::new().induction_hypothesis("property holds for k states");

        // Chain another method
        let builder = builder.vc("test");
        let proof = builder.build();

        // Check that the induction_hypothesis step was added
        // SmtStep::Assume has 'name' field
        let has_hyp = proof.steps.iter().any(|s| {
            if let ProofStep::Smt(SmtStep::Assume { name, .. }) = s {
                name == "induction_hypothesis"
            } else {
                false
            }
        });
        assert!(has_hyp);
    }

    #[test]
    fn test_builder_induction_step_returns_self() {
        // Line 229: induction_step() returns Self, not Default::default()
        let builder =
            KInductionProofBuilder::new().induction_step(vec![0, 1, 2], "property holds at k+1");

        // Chain another method
        let builder = builder.vc("test");
        let proof = builder.build();

        // Check that the induction_step was added
        // SmtStep::Inference has 'rule' field
        let has_step = proof.steps.iter().any(|s| {
            if let ProofStep::Smt(SmtStep::Inference { rule, .. }) = s {
                rule == "induction_step"
            } else {
                false
            }
        });
        assert!(has_step);
    }

    #[test]
    fn test_builder_chaining_preserves_all_values() {
        // Comprehensive test that all builder methods return Self properly
        let proof = KInductionProofBuilder::new()
            .vc("(assert P)")
            .k(10)
            .base_case(0, "P(0)")
            .base_case(1, "P(1)")
            .induction_hypothesis("forall k. P(k)")
            .induction_step(vec![0, 1], "P(k+1)")
            .invariant("(>= x 0)")
            .property_name("test_property")
            .description("Test description")
            .generation_time(Duration::from_millis(100))
            .build();

        // Verify all values were preserved
        assert!(proof.vc.contains("(assert P)"));
        assert_eq!(
            proof.metadata.property_name,
            Some("test_property".to_string())
        );
        assert_eq!(
            proof.metadata.description,
            Some("Test description".to_string())
        );

        // Count proof steps - SmtStep::Assume has 'name' field
        let base_case_count = proof
            .steps
            .iter()
            .filter(|s| {
                if let ProofStep::Smt(SmtStep::Assume { name, .. }) = s {
                    name.starts_with("base_case_")
                } else {
                    false
                }
            })
            .count();
        assert_eq!(base_case_count, 2);
    }

    #[test]
    fn test_builder_default_vs_new() {
        // Verify Default::default() and new() produce identical builders
        let default_proof = KInductionProofBuilder::default().build();
        let new_proof = KInductionProofBuilder::new().build();

        // Both should have same backend and format
        assert_eq!(default_proof.backend, new_proof.backend);
        assert_eq!(default_proof.format, new_proof.format);
    }

    #[test]
    fn test_builder_without_optional_fields() {
        // Test builder without property_name and description
        let proof = KInductionProofBuilder::new()
            .vc("(assert true)")
            .k(1)
            .build();

        // Should build successfully without optional fields
        assert!(proof.metadata.property_name.is_none());
        assert!(proof.metadata.description.is_none());
    }

    #[test]
    fn test_builder_base_case_format() {
        // Verify base_case creates proper ProofStep::Smt with correct name
        let proof = KInductionProofBuilder::new()
            .base_case(42, "formula_42")
            .build();

        let found = proof.steps.iter().find(|s| {
            if let ProofStep::Smt(SmtStep::Assume { name, .. }) = s {
                name == "base_case_42"
            } else {
                false
            }
        });
        assert!(found.is_some());
    }
}
