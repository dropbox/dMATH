use dashprove_ai::repair::RepairKind;
use dashprove_ai::{Confidence, ExplanationKind};
use dashprove_ai::{ProofAssistant, ProofSketch, StrategyModel, SuggestionSource};
use dashprove_backends::traits::{BackendId, VerificationStatus};
use dashprove_learning::{LearnableResult, ProofLearningSystem};
use dashprove_usl::ast::{Contract, Expr, Param, Property, Temporal, TemporalExpr, Theorem, Type};
use serde_json::{json, Value};
use std::time::Duration;

fn make_theorem(name: &str) -> Property {
    Property::Theorem(Theorem {
        name: name.to_string(),
        body: Expr::ForAll {
            var: "x".to_string(),
            ty: Some(Type::Named("Bool".to_string())),
            body: Box::new(Expr::Or(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Not(Box::new(Expr::Var("x".to_string())))),
            )),
        },
    })
}

fn make_contract(name: &str) -> Property {
    Property::Contract(Contract {
        type_path: vec![name.to_string()],
        params: vec![Param {
            name: "x".to_string(),
            ty: Type::Named("i32".to_string()),
        }],
        return_type: Some(Type::Named("i32".to_string())),
        requires: vec![Expr::Compare(
            Box::new(Expr::Var("x".to_string())),
            dashprove_usl::ast::ComparisonOp::Ge,
            Box::new(Expr::Int(0)),
        )],
        ensures: vec![Expr::Compare(
            Box::new(Expr::Var("x".to_string())),
            dashprove_usl::ast::ComparisonOp::Ge,
            Box::new(Expr::Int(0)),
        )],
        ensures_err: vec![],
        assigns: vec![],
        allocates: vec![],
        frees: vec![],
        terminates: None,
        decreases: None,
        behaviors: vec![],
        complete_behaviors: false,
        disjoint_behaviors: false,
    })
}

fn make_temporal(name: &str) -> Property {
    Property::Temporal(Temporal {
        name: name.to_string(),
        body: TemporalExpr::Always(Box::new(TemporalExpr::Atom(Expr::Bool(true)))),
        fairness: vec![],
    })
}

fn backend_to_index(backend: BackendId) -> usize {
    match backend {
        BackendId::Lean4 => 0,
        BackendId::TlaPlus => 1,
        BackendId::Apalache => 21,
        BackendId::Kani => 2,
        BackendId::Alloy => 3,
        BackendId::Isabelle => 4,
        BackendId::Coq => 5,
        BackendId::Dafny => 6,
        BackendId::Marabou => 7,
        BackendId::AlphaBetaCrown => 8,
        BackendId::Eran => 9,
        BackendId::Storm => 10,
        BackendId::Prism => 11,
        BackendId::Tamarin => 12,
        BackendId::ProVerif => 13,
        BackendId::Verifpal => 14,
        BackendId::Verus => 15,
        BackendId::Creusot => 16,
        BackendId::Prusti => 17,
        BackendId::Z3 => 18,
        BackendId::Cvc5 => 19,
        BackendId::PlatformApi => 20,
        // New backends (Phase 12) get default index
        _ => 22,
    }
}

fn make_biased_model(target_backend: BackendId, tactic_name: &str) -> StrategyModel {
    // Serialize, edit biases for deterministic predictions, then deserialize.
    let mut predictor_json =
        serde_json::to_value(dashprove_ai::StrategyPredictor::new()).expect("serialize predictor");

    let tactic_names: Vec<String> = predictor_json
        .get("tactic_names")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let tactic_idx = tactic_names
        .iter()
        .position(|name| name == tactic_name)
        .unwrap_or(0);
    let tactic_count = tactic_names.len().max(1);

    if let Some(obj) = predictor_json.as_object_mut() {
        if let Some(backend_output) = obj.get_mut("backend_output").and_then(Value::as_object_mut) {
            if let Some(weights) = backend_output
                .get_mut("weights")
                .and_then(Value::as_array_mut)
            {
                for row in weights {
                    if let Some(entries) = row.as_array_mut() {
                        for weight in entries {
                            *weight = json!(0.0);
                        }
                    }
                }
            }
            if let Some(biases) = backend_output
                .get_mut("biases")
                .and_then(Value::as_array_mut)
            {
                for bias in biases.iter_mut() {
                    *bias = json!(0.0);
                }
                let idx = backend_to_index(target_backend);
                if let Some(entry) = biases.get_mut(idx) {
                    // Bias needs to be high enough to achieve 0.8 confidence after softmax
                    // With 200+ backends: exp(b)/(exp(b)+200) >= 0.8 requires b >= 7.0
                    *entry = json!(6.5);
                }
            }
        }

        if let Some(tactic_output) = obj.get_mut("tactic_output").and_then(Value::as_object_mut) {
            if let Some(weights) = tactic_output
                .get_mut("weights")
                .and_then(Value::as_array_mut)
            {
                for row in weights {
                    if let Some(entries) = row.as_array_mut() {
                        for weight in entries {
                            *weight = json!(0.0);
                        }
                    }
                }
            }
            if let Some(biases) = tactic_output
                .get_mut("biases")
                .and_then(Value::as_array_mut)
            {
                for bias in biases.iter_mut() {
                    *bias = json!(0.0);
                }
                let slot = tactic_idx;
                if let Some(entry) = biases.get_mut(slot) {
                    *entry = json!(4.0);
                }
                let second_slot = tactic_idx + tactic_count;
                if let Some(entry) = biases.get_mut(second_slot) {
                    *entry = json!(3.0);
                }
            }
        }
    }

    let predictor =
        serde_json::from_value::<dashprove_ai::StrategyPredictor>(predictor_json).unwrap();
    StrategyModel::from(predictor)
}

#[test]
fn proof_assistant_uses_learning_corpus() {
    let mut learning = ProofLearningSystem::new();
    let property = make_theorem("learning_example");

    let learnable = LearnableResult {
        property: property.clone(),
        backend: BackendId::Lean4,
        status: VerificationStatus::Proven,
        tactics: vec!["simp".to_string(), "intro".to_string()],
        time_taken: Duration::from_millis(25),
        proof_output: Some("theorem learning_example : True := by decide".to_string()),
    };
    learning.record(&learnable);

    let assistant = ProofAssistant::with_learning(learning);
    let strategy = assistant.recommend_strategy(&property);

    assert_ne!(strategy.confidence, Confidence::Speculative);
    assert!(strategy
        .tactics
        .iter()
        .any(|t| t.source == SuggestionSource::Learning));
    assert!(strategy
        .tactics
        .iter()
        .any(|t| t.tactic == "simp" || t.tactic == "intro"));
}

#[test]
fn ml_prediction_is_used_when_available() {
    let model = make_biased_model(BackendId::Kani, "bounded_check");
    let assistant = ProofAssistant::with_ml_predictor(model);
    let property = make_contract("safe_add");

    let strategy = assistant.recommend_strategy(&property);

    assert_eq!(strategy.backend, BackendId::Kani);
    assert_eq!(strategy.confidence, Confidence::High);
    assert!(strategy.rationale.contains("ML model predicts"));
    assert!(strategy
        .tactics
        .iter()
        .any(|t| t.tactic == "bounded_check" && t.source == SuggestionSource::Learning));
}

#[test]
fn sketch_generation_keeps_hints_and_targets_backend() {
    let assistant = ProofAssistant::new();
    let property = make_temporal("always_safe");
    let hints = vec!["fairness".to_string(), "use temporal induction".to_string()];

    let sketch: ProofSketch = assistant.create_sketch(&property, &hints);
    let rendered = sketch.to_lean();

    assert_eq!(sketch.target_backend, BackendId::TlaPlus);
    assert!(sketch.hints_used.contains(&"fairness".to_string()));
    assert!(!sketch.steps.is_empty());
    assert!(rendered.contains("Goal"));
}

#[test]
fn counterexample_and_repair_helpers_are_exposed() {
    let assistant = ProofAssistant::new();
    let property = make_temporal("safety_violation");
    let trace_output = r#"
State 1:
  x = 0
State 2:
  /\ Next_action
  x = 1
"#;

    let explanation =
        assistant.explain_counterexample(&property, trace_output, &BackendId::TlaPlus);
    assert_eq!(explanation.kind, ExplanationKind::StateTrace);
    assert!(!explanation.trace.is_empty());
    assert!(explanation.summary.contains("violated"));

    let lean_error = "unknown identifier 'missing_lemma'";
    let repairs =
        assistant.suggest_repairs(&make_theorem("broken"), None, lean_error, &BackendId::Lean4);

    assert!(!repairs.is_empty());
    assert_eq!(repairs[0].kind, RepairKind::AddImport);
    assert!(repairs[0].fix.contains("import"));
}
