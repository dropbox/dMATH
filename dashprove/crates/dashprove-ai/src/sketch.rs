//! Proof sketching system
//!
//! Proof sketches provide a high-level outline of a proof that can be
//! elaborated by the verification backend. This allows AI to provide
//! strategic guidance while the backend fills in tactical details.

use dashprove_backends::traits::BackendId;
use dashprove_learning::ProofLearningSystem;
use dashprove_usl::ast::{Expr, Property};
use serde::{Deserialize, Serialize};

/// A step in a proof sketch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SketchStep {
    /// The goal at this step
    pub goal: String,
    /// Strategy or tactic hint
    pub strategy: String,
    /// Whether this step is complete or needs elaboration
    pub complete: bool,
    /// Sub-steps if this is a compound step
    pub substeps: Vec<SketchStep>,
}

/// A proof sketch - high level proof outline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofSketch {
    /// Name of the property being proved
    pub property_name: String,
    /// Top-level steps in the proof
    pub steps: Vec<SketchStep>,
    /// Backend this sketch targets
    pub target_backend: BackendId,
    /// User-provided hints incorporated
    pub hints_used: Vec<String>,
}

impl ProofSketch {
    /// Create an empty sketch
    pub fn empty(property_name: &str, backend: BackendId) -> Self {
        Self {
            property_name: property_name.to_string(),
            steps: vec![],
            target_backend: backend,
            hints_used: vec![],
        }
    }

    /// Add a step to the sketch
    pub fn add_step(&mut self, goal: &str, strategy: &str) {
        self.steps.push(SketchStep {
            goal: goal.to_string(),
            strategy: strategy.to_string(),
            complete: false,
            substeps: vec![],
        });
    }

    /// Render the sketch as LEAN 4 tactics
    pub fn to_lean(&self) -> String {
        let mut output = String::new();
        for step in &self.steps {
            output.push_str(&step_to_lean(step, 0));
        }
        output
    }

    /// Check if the sketch is complete (all steps filled)
    pub fn is_complete(&self) -> bool {
        self.steps
            .iter()
            .all(|s| s.complete && substeps_complete(s))
    }
}

fn substeps_complete(step: &SketchStep) -> bool {
    step.substeps
        .iter()
        .all(|s| s.complete && substeps_complete(s))
}

fn step_to_lean(step: &SketchStep, indent: usize) -> String {
    let prefix = "  ".repeat(indent);
    let mut output = String::new();

    if step.complete {
        output.push_str(&format!("{}{}\n", prefix, step.strategy));
    } else {
        output.push_str(&format!("{}-- Goal: {}\n", prefix, step.goal));
        output.push_str(&format!("{}sorry -- {}\n", prefix, step.strategy));
    }

    for substep in &step.substeps {
        output.push_str(&step_to_lean(substep, indent + 1));
    }

    output
}

/// Create a proof sketch from a property and hints
pub fn create_sketch(
    property: &Property,
    hints: &[String],
    learning: Option<&ProofLearningSystem>,
) -> ProofSketch {
    let (name, backend) = match property {
        Property::Theorem(t) => (&t.name, BackendId::Lean4),
        Property::Invariant(i) => (&i.name, BackendId::Lean4),
        Property::Temporal(t) => (&t.name, BackendId::TlaPlus),
        Property::Contract(c) => (&c.type_path.join("::"), BackendId::Kani),
        Property::Refinement(r) => (&r.name, BackendId::Lean4),
        Property::Probabilistic(p) => (&p.name, BackendId::Lean4),
        Property::Security(s) => (&s.name, BackendId::Lean4),
        Property::Semantic(s) => (&s.name, BackendId::Lean4),
        Property::PlatformApi(p) => (&p.name, BackendId::PlatformApi),
        Property::Bisimulation(b) => (&b.name, BackendId::Kani), // Use Kani for bisim as default
        Property::Version(v) => (&v.name, BackendId::Lean4),     // Version specs use Lean4
        Property::Capability(c) => (&c.name, BackendId::Lean4),  // Capability specs use Lean4
        Property::DistributedInvariant(d) => (&d.name, BackendId::TlaPlus), // Distributed invariants use TLA+
        Property::DistributedTemporal(d) => (&d.name, BackendId::TlaPlus), // Distributed temporal uses TLA+
        Property::Composed(c) => (&c.name, BackendId::Lean4), // Composed theorems use Lean4
        Property::ImprovementProposal(i) => (&i.name, BackendId::Lean4), // Improvement proposals use Lean4
        Property::VerificationGate(v) => (&v.name, BackendId::Lean4), // Verification gates use Lean4
        Property::Rollback(r) => (&r.name, BackendId::TlaPlus),       // Rollback specs use TLA+
    };

    let mut sketch = ProofSketch::empty(name, backend);

    // Incorporate user hints
    sketch.hints_used = hints.to_vec();

    // Generate steps based on property structure
    match property {
        Property::Theorem(t) => {
            generate_theorem_steps(&mut sketch, &t.body, learning);
        }
        Property::Invariant(i) => {
            generate_invariant_steps(&mut sketch, &i.body);
        }
        Property::Semantic(s) => {
            generate_theorem_steps(&mut sketch, &s.body, learning);
        }
        Property::Temporal(t) => {
            generate_temporal_steps(&mut sketch, &t.body);
        }
        Property::Contract(c) => {
            generate_contract_steps(&mut sketch, c);
        }
        Property::Refinement(r) => {
            generate_refinement_steps(&mut sketch, r);
        }
        Property::DistributedInvariant(d) => {
            // Distributed invariant - like invariant but with multi-agent semantics
            generate_invariant_steps(&mut sketch, &d.body);
        }
        Property::DistributedTemporal(d) => {
            // Distributed temporal - like temporal but with multi-agent semantics
            generate_temporal_steps(&mut sketch, &d.body);
        }
        _ => {
            sketch.add_step("Prove property", "Apply appropriate strategy");
        }
    }

    sketch
}

/// Generate sketch steps for a theorem
fn generate_theorem_steps(
    sketch: &mut ProofSketch,
    body: &Expr,
    learning: Option<&ProofLearningSystem>,
) {
    match body {
        Expr::ForAll {
            var, body: inner, ..
        } => {
            sketch.add_step(&format!("Introduce {}", var), &format!("intro {}", var));
            generate_theorem_steps(sketch, inner, learning);
        }
        Expr::Exists {
            var, body: inner, ..
        } => {
            sketch.add_step(&format!("Provide witness for {}", var), "use <witness>");
            generate_theorem_steps(sketch, inner, learning);
        }
        Expr::Implies(_hyp, concl) => {
            sketch.add_step("Introduce hypothesis", "intro h");
            // Check if conclusion suggests specific tactics
            let tactic = suggest_for_conclusion(concl, learning);
            sketch.add_step("Prove conclusion using hypothesis", &tactic);
        }
        Expr::And(lhs, rhs) => {
            sketch.add_step("Split conjunction", "constructor");
            let mut left_sketch = ProofSketch::empty(&sketch.property_name, sketch.target_backend);
            generate_theorem_steps(&mut left_sketch, lhs, learning);
            if let Some(step) = sketch.steps.last_mut() {
                step.substeps.extend(left_sketch.steps);
            }

            let mut right_sketch = ProofSketch::empty(&sketch.property_name, sketch.target_backend);
            generate_theorem_steps(&mut right_sketch, rhs, learning);
            if let Some(step) = sketch.steps.last_mut() {
                step.substeps.extend(right_sketch.steps);
            }
        }
        Expr::Or(_, _) => {
            sketch.add_step("Prove disjunction", "left -- or right");
        }
        Expr::Not(inner) => {
            sketch.add_step("Prove negation", "intro h");
            generate_theorem_steps(sketch, inner, learning);
        }
        Expr::Compare(_, _, _) => {
            sketch.add_step("Prove comparison", "omega");
        }
        Expr::Binary(_, _, _) => {
            sketch.add_step("Simplify arithmetic", "ring");
        }
        Expr::Bool(true) => {
            sketch.add_step("Prove trivial goal", "trivial");
            if let Some(step) = sketch.steps.last_mut() {
                step.complete = true;
            }
        }
        Expr::App(name, _) => {
            sketch.add_step(
                &format!("Unfold and simplify {}", name),
                &format!("simp [{}]", name),
            );
        }
        _ => {
            sketch.add_step("Complete proof", "sorry");
        }
    }
}

/// Generate sketch steps for an invariant
fn generate_invariant_steps(sketch: &mut ProofSketch, body: &Expr) {
    sketch.add_step("Prove invariant holds", "decide");

    if matches!(body, Expr::ForAll { .. }) {
        sketch.add_step("Introduce quantified variable", "intro");
    }
}

/// Generate sketch steps for temporal property
fn generate_temporal_steps(sketch: &mut ProofSketch, body: &dashprove_usl::ast::TemporalExpr) {
    use dashprove_usl::ast::TemporalExpr;

    match body {
        TemporalExpr::Always(inner) => {
            sketch.add_step("Prove invariant by induction", "InductiveInvariant");
            sketch.add_step("Base case: Init => P", "InitCase");
            sketch.add_step("Inductive case: P /\\ Next => P'", "InductiveCase");
            generate_temporal_steps(sketch, inner);
        }
        TemporalExpr::Eventually(inner) => {
            sketch.add_step("Prove liveness", "LivenessArgument");
            sketch.add_step("Show progress is always possible", "FairnessAssumption");
            generate_temporal_steps(sketch, inner);
        }
        TemporalExpr::LeadsTo(from, to) => {
            sketch.add_step("Prove leads-to", "LeadsToArgument");
            sketch.add_step("Establish ranking function", "WellFoundedOrder");
            generate_temporal_steps(sketch, from);
            generate_temporal_steps(sketch, to);
        }
        TemporalExpr::Atom(_) => {
            sketch.add_step("Prove state predicate", "StatePredicate");
        }
    }
}

/// Generate sketch steps for contract verification
fn generate_contract_steps(sketch: &mut ProofSketch, contract: &dashprove_usl::ast::Contract) {
    if !contract.requires.is_empty() {
        sketch.add_step("Assume preconditions", "kani::assume(preconditions)");
    }

    sketch.add_step("Execute function under test", "let result = f(...)");

    if !contract.ensures.is_empty() {
        sketch.add_step("Verify postconditions", "kani::assert(postconditions)");
    }

    if !contract.ensures_err.is_empty() {
        sketch.add_step(
            "Verify error postconditions",
            "kani::assert(error_postconditions)",
        );
    }
}

/// Generate sketch steps for refinement proof
fn generate_refinement_steps(
    sketch: &mut ProofSketch,
    refinement: &dashprove_usl::ast::Refinement,
) {
    sketch.add_step(
        &format!("Prove {} refines {}", refinement.name, refinement.refines),
        "RefinementProof",
    );
    sketch.add_step("Prove abstraction relation", "AbstractionProof");
    sketch.add_step("Prove forward simulation", "SimulationProof");
}

/// Suggest a tactic for a conclusion based on structure and learning
fn suggest_for_conclusion(expr: &Expr, learning: Option<&ProofLearningSystem>) -> String {
    // First check learning system
    if let Some(ls) = learning {
        // Create a temporary property for suggestion lookup
        let temp_prop = Property::Theorem(dashprove_usl::ast::Theorem {
            name: "temp".to_string(),
            body: expr.clone(),
        });
        let suggestions = ls.suggest_tactics(&temp_prop, 1);
        if let Some((tactic, _)) = suggestions.first() {
            return tactic.clone();
        }
    }

    // Fall back to structural analysis
    match expr {
        Expr::Compare(_, _, _) => "omega".to_string(),
        Expr::Binary(_, _, _) => "ring".to_string(),
        Expr::Bool(true) => "trivial".to_string(),
        Expr::And(_, _) => "constructor".to_string(),
        Expr::Or(_, _) => "left -- or right".to_string(),
        Expr::App(name, _) => format!("simp [{}]", name),
        _ => "exact <proof>".to_string(),
    }
}

/// Elaborate a sketch into concrete tactics (placeholder for future AI elaboration)
pub fn elaborate_sketch(sketch: &ProofSketch) -> Result<String, crate::AiError> {
    // For now, just convert to LEAN
    // Future: use AI to fill in holes
    Ok(sketch.to_lean())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashprove_usl::ast::{Invariant, Temporal, TemporalExpr, Theorem, Type};

    #[test]
    fn test_empty_sketch() {
        let sketch = ProofSketch::empty("test", BackendId::Lean4);
        assert!(sketch.steps.is_empty());
        assert_eq!(sketch.property_name, "test");
    }

    #[test]
    fn test_sketch_add_step() {
        let mut sketch = ProofSketch::empty("test", BackendId::Lean4);
        sketch.add_step("Introduce x", "intro x");

        assert_eq!(sketch.steps.len(), 1);
        assert_eq!(sketch.steps[0].goal, "Introduce x");
        assert_eq!(sketch.steps[0].strategy, "intro x");
    }

    #[test]
    fn test_forall_sketch() {
        let prop = Property::Theorem(Theorem {
            name: "forall_test".to_string(),
            body: Expr::ForAll {
                var: "x".to_string(),
                ty: Some(Type::Named("Nat".to_string())),
                body: Box::new(Expr::Bool(true)),
            },
        });

        let sketch = create_sketch(&prop, &[], None);

        assert!(!sketch.steps.is_empty());
        assert!(sketch.steps[0].strategy.contains("intro"));
    }

    #[test]
    fn test_implies_sketch() {
        let prop = Property::Theorem(Theorem {
            name: "implies_test".to_string(),
            body: Expr::Implies(
                Box::new(Expr::Var("P".to_string())),
                Box::new(Expr::Var("Q".to_string())),
            ),
        });

        let sketch = create_sketch(&prop, &[], None);

        assert!(sketch.steps.iter().any(|s| s.strategy.contains("intro")));
    }

    #[test]
    fn test_temporal_always_sketch() {
        let prop = Property::Temporal(Temporal {
            name: "always_test".to_string(),
            body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
            fairness: vec![],
        });

        let sketch = create_sketch(&prop, &[], None);

        assert!(sketch.steps.iter().any(|s| s.goal.contains("induction")));
        assert!(sketch.steps.iter().any(|s| s.goal.contains("Base case")));
    }

    #[test]
    fn test_sketch_to_lean() {
        let mut sketch = ProofSketch::empty("test", BackendId::Lean4);
        sketch.add_step("Introduce x", "intro x");
        sketch.add_step("Prove goal", "decide");

        let lean = sketch.to_lean();
        assert!(lean.contains("-- Goal: Introduce x"));
        assert!(lean.contains("sorry -- intro x"));
    }

    #[test]
    fn test_complete_step() {
        let mut sketch = ProofSketch::empty("test", BackendId::Lean4);
        sketch.steps.push(SketchStep {
            goal: "Trivial".to_string(),
            strategy: "trivial".to_string(),
            complete: true,
            substeps: vec![],
        });

        let lean = sketch.to_lean();
        assert!(lean.contains("trivial"));
        assert!(!lean.contains("sorry"));
    }

    #[test]
    fn test_invariant_sketch() {
        let prop = Property::Invariant(Invariant {
            name: "test_inv".to_string(),
            body: Expr::Bool(true),
        });

        let sketch = create_sketch(&prop, &[], None);

        assert!(!sketch.steps.is_empty());
    }

    #[test]
    fn test_elaborate_sketch() {
        let mut sketch = ProofSketch::empty("test", BackendId::Lean4);
        sketch.add_step("Intro", "intro x");

        let result = elaborate_sketch(&sketch);
        assert!(result.is_ok());
    }
}

// ========== Kani proof harnesses ==========

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify SketchStep stores goal correctly
    #[kani::proof]
    fn verify_sketch_step_stores_goal() {
        let step = SketchStep {
            goal: String::from("test goal"),
            strategy: String::from("intro"),
            complete: false,
            substeps: vec![],
        };
        kani::assert(!step.goal.is_empty(), "goal should be stored");
    }

    /// Verify SketchStep stores strategy correctly
    #[kani::proof]
    fn verify_sketch_step_stores_strategy() {
        let step = SketchStep {
            goal: String::from("test goal"),
            strategy: String::from("intro"),
            complete: false,
            substeps: vec![],
        };
        kani::assert(!step.strategy.is_empty(), "strategy should be stored");
    }

    /// Verify SketchStep complete flag
    #[kani::proof]
    fn verify_sketch_step_complete_false() {
        let step = SketchStep {
            goal: String::from("test"),
            strategy: String::from("sorry"),
            complete: false,
            substeps: vec![],
        };
        kani::assert(!step.complete, "complete should be false");
    }

    /// Verify SketchStep complete flag true
    #[kani::proof]
    fn verify_sketch_step_complete_true() {
        let step = SketchStep {
            goal: String::from("test"),
            strategy: String::from("trivial"),
            complete: true,
            substeps: vec![],
        };
        kani::assert(step.complete, "complete should be true");
    }

    /// Verify SketchStep empty substeps
    #[kani::proof]
    fn verify_sketch_step_empty_substeps() {
        let step = SketchStep {
            goal: String::from("test"),
            strategy: String::from("intro"),
            complete: false,
            substeps: vec![],
        };
        kani::assert(step.substeps.is_empty(), "substeps should be empty");
    }

    /// Verify ProofSketch::empty creates empty sketch
    #[kani::proof]
    fn verify_proof_sketch_empty() {
        let sketch = ProofSketch::empty("test_prop", BackendId::Lean4);
        kani::assert(sketch.steps.is_empty(), "steps should be empty");
        kani::assert(sketch.hints_used.is_empty(), "hints_used should be empty");
    }

    /// Verify ProofSketch::empty stores property name
    #[kani::proof]
    fn verify_proof_sketch_stores_name() {
        let sketch = ProofSketch::empty("my_property", BackendId::Lean4);
        kani::assert(
            !sketch.property_name.is_empty(),
            "property_name should be stored",
        );
    }

    /// Verify ProofSketch::empty stores backend
    #[kani::proof]
    fn verify_proof_sketch_stores_backend_lean4() {
        let sketch = ProofSketch::empty("test", BackendId::Lean4);
        kani::assert(
            matches!(sketch.target_backend, BackendId::Lean4),
            "target_backend should be Lean4",
        );
    }

    /// Verify ProofSketch::empty stores backend TlaPlus
    #[kani::proof]
    fn verify_proof_sketch_stores_backend_tlaplus() {
        let sketch = ProofSketch::empty("test", BackendId::TlaPlus);
        kani::assert(
            matches!(sketch.target_backend, BackendId::TlaPlus),
            "target_backend should be TlaPlus",
        );
    }

    /// Verify ProofSketch::empty stores backend Kani
    #[kani::proof]
    fn verify_proof_sketch_stores_backend_kani() {
        let sketch = ProofSketch::empty("test", BackendId::Kani);
        kani::assert(
            matches!(sketch.target_backend, BackendId::Kani),
            "target_backend should be Kani",
        );
    }

    /// Verify ProofSketch::add_step adds a step
    #[kani::proof]
    fn verify_proof_sketch_add_step() {
        let mut sketch = ProofSketch::empty("test", BackendId::Lean4);
        sketch.add_step("Introduce x", "intro x");
        kani::assert(sketch.steps.len() == 1, "should have one step");
    }

    /// Verify ProofSketch::add_step stores goal
    #[kani::proof]
    fn verify_proof_sketch_add_step_goal() {
        let mut sketch = ProofSketch::empty("test", BackendId::Lean4);
        sketch.add_step("Introduce x", "intro x");
        kani::assert(!sketch.steps[0].goal.is_empty(), "goal should be stored");
    }

    /// Verify ProofSketch::add_step stores strategy
    #[kani::proof]
    fn verify_proof_sketch_add_step_strategy() {
        let mut sketch = ProofSketch::empty("test", BackendId::Lean4);
        sketch.add_step("Introduce x", "intro x");
        kani::assert(
            !sketch.steps[0].strategy.is_empty(),
            "strategy should be stored",
        );
    }

    /// Verify ProofSketch::add_step sets complete to false
    #[kani::proof]
    fn verify_proof_sketch_add_step_incomplete() {
        let mut sketch = ProofSketch::empty("test", BackendId::Lean4);
        sketch.add_step("Introduce x", "intro x");
        kani::assert(!sketch.steps[0].complete, "added step should be incomplete");
    }

    /// Verify ProofSketch::is_complete on empty sketch
    #[kani::proof]
    fn verify_proof_sketch_empty_is_complete() {
        let sketch = ProofSketch::empty("test", BackendId::Lean4);
        kani::assert(sketch.is_complete(), "empty sketch should be complete");
    }

    /// Verify substeps_complete on step with no substeps
    #[kani::proof]
    fn verify_substeps_complete_empty() {
        let step = SketchStep {
            goal: String::from("test"),
            strategy: String::from("intro"),
            complete: true,
            substeps: vec![],
        };
        kani::assert(
            substeps_complete(&step),
            "empty substeps should be complete",
        );
    }
}
