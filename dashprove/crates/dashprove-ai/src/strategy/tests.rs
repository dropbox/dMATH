//! Tests for the strategy prediction module

use super::*;
use dashprove_usl::ast::{ComparisonOp, Expr, Invariant, Temporal, TemporalExpr, Theorem};

fn make_theorem(name: &str) -> Property {
    Property::Theorem(Theorem {
        name: name.to_string(),
        body: Expr::ForAll {
            var: "x".to_string(),
            ty: Some(dashprove_usl::ast::Type::Named("Bool".to_string())),
            body: Box::new(Expr::Or(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Not(Box::new(Expr::Var("x".to_string())))),
            )),
        },
    })
}

fn make_temporal(name: &str) -> Property {
    Property::Temporal(Temporal {
        name: name.to_string(),
        body: TemporalExpr::Always(Box::new(TemporalExpr::Eventually(Box::new(
            TemporalExpr::Atom(Expr::Var("ready".to_string())),
        )))),
        fairness: vec![],
    })
}

fn make_invariant(name: &str) -> Property {
    Property::Invariant(Invariant {
        name: name.to_string(),
        body: Expr::ForAll {
            var: "n".to_string(),
            ty: Some(dashprove_usl::ast::Type::Named("Nat".to_string())),
            body: Box::new(Expr::Compare(
                Box::new(Expr::Var("n".to_string())),
                ComparisonOp::Ge,
                Box::new(Expr::Int(0)),
            )),
        },
    })
}

fn make_biased_predictor(target: BackendId) -> StrategyPredictor {
    let mut predictor = StrategyPredictor::new();

    // Zero out weights so biases dominate
    for row in predictor.backend_output.weights.iter_mut() {
        for w in row.iter_mut() {
            *w = 0.0;
        }
    }
    for bias in predictor.backend_output.biases.iter_mut() {
        *bias = 0.0;
    }

    let idx = backend_to_idx(target);
    if idx < predictor.backend_output.biases.len() {
        predictor.backend_output.biases[idx] = 5.0;
    }

    predictor
}

#[test]
fn test_feature_extraction_theorem() {
    let prop = make_theorem("test_lem");
    let features = PropertyFeatureVector::from_property(&prop);

    assert_eq!(features.features.len(), NUM_FEATURES);
    assert_eq!(features.get("prop_theorem"), Some(1.0));
    assert_eq!(features.get("prop_temporal"), Some(0.0));
    assert!(features.get("forall_count").unwrap() > 0.0);
}

#[test]
fn test_feature_extraction_temporal() {
    let prop = make_temporal("liveness");
    let features = PropertyFeatureVector::from_property(&prop);

    assert_eq!(features.get("prop_temporal"), Some(1.0));
    assert!(features.get("temporal_always").unwrap() > 0.0);
    assert!(features.get("temporal_eventually").unwrap() > 0.0);
}

#[test]
fn test_feature_extraction_invariant() {
    let prop = make_invariant("positive");
    let features = PropertyFeatureVector::from_property(&prop);

    assert_eq!(features.get("prop_invariant"), Some(1.0));
    assert!(features.get("forall_count").unwrap() > 0.0);
    assert!(features.get("comparison_ops").unwrap() > 0.0);
}

#[test]
fn test_predictor_creation() {
    let predictor = StrategyPredictor::new();
    assert!(!predictor.tactic_names.is_empty());
}

#[test]
fn test_backend_prediction() {
    let predictor = StrategyPredictor::new();
    let prop = make_theorem("test");
    let features = PropertyFeatureVector::from_property(&prop);

    let prediction = predictor.predict_backend(&features);

    // Confidence should be valid probability
    assert!(prediction.confidence >= 0.0);
    assert!(prediction.confidence <= 1.0);

    // Should have alternatives
    assert!(!prediction.alternatives.is_empty());
}

#[test]
fn test_backend_probabilities_are_normalized() {
    let predictor = StrategyPredictor::new();
    let prop = make_theorem("probabilities");
    let features = PropertyFeatureVector::from_property(&prop);

    let probs = predictor.backend_probabilities(&features);
    let sum: f64 = probs.iter().map(|(_, p)| *p).sum();

    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_ensemble_prefers_higher_weight() {
    let prop = make_theorem("ensemble_pref");
    let features = PropertyFeatureVector::from_property(&prop);

    let lean = EnsembleMember::new(make_biased_predictor(BackendId::Lean4)).with_weight(0.25);
    let kani = EnsembleMember::new(make_biased_predictor(BackendId::Kani)).with_weight(1.25);

    let ensemble = EnsembleStrategyPredictor::new(vec![lean, kani]);
    let prediction = ensemble.predict_backend(&features);

    // Kani should be chosen due to higher weight
    assert_eq!(prediction.backend, BackendId::Kani);
    // Confidence should be well above random baseline (1/NUM_BACKENDS ≈ 0.005)
    // With 200+ backends and weighted voting, expect confidence > 0.3
    assert!(
        prediction.confidence > 0.3,
        "Expected confidence > 0.3, got {}",
        prediction.confidence
    );
}

#[test]
fn test_strategy_model_loads_legacy_single() {
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let path = dir.path().join("legacy_model.json");

    StrategyPredictor::new().save(&path).unwrap();

    let model = StrategyModel::load(&path).unwrap();
    match model {
        StrategyModel::Single { .. } => {}
        StrategyModel::Ensemble { .. } => panic!("expected single model"),
    }
}

#[test]
fn test_tactic_prediction() {
    let predictor = StrategyPredictor::new();
    let prop = make_theorem("test");
    let features = PropertyFeatureVector::from_property(&prop);

    let predictions = predictor.predict_tactics(&features, 5);

    // Tactics should have valid confidence and position
    for tactic in &predictions {
        assert!(tactic.confidence >= 0.0);
        assert!(tactic.confidence <= 1.0);
        assert!(tactic.position < MAX_TACTICS);
    }
}

#[test]
fn test_time_prediction() {
    let predictor = StrategyPredictor::new();
    let prop = make_theorem("test");
    let features = PropertyFeatureVector::from_property(&prop);

    let prediction = predictor.predict_time(&features);

    // Time should be positive and reasonable
    assert!(prediction.expected_seconds > 0.0);
    assert!(prediction.expected_seconds <= 3600.0);
    assert!(prediction.range.0 <= prediction.expected_seconds);
    assert!(prediction.range.1 >= prediction.expected_seconds);
}

#[test]
fn test_full_strategy_prediction() {
    let predictor = StrategyPredictor::new();
    let prop = make_theorem("test");

    let strategy = predictor.predict_strategy(&prop);

    assert!(strategy.backend.confidence > 0.0);
    assert!(strategy.time.expected_seconds > 0.0);
    assert!(!strategy.features.features.is_empty());
}

#[test]
fn test_training_data_generator() {
    let mut generator = TrainingDataGenerator::new();

    let prop1 = make_theorem("thm1");
    let prop2 = make_temporal("temp1");

    generator.add_success(&prop1, BackendId::Lean4, vec!["simp".to_string()], 1.5);
    generator.add_success(
        &prop2,
        BackendId::TlaPlus,
        vec!["model_check".to_string()],
        5.0,
    );
    generator.add_failure(&prop1, BackendId::Coq, vec!["auto".to_string()], 10.0);

    let stats = generator.stats();
    assert_eq!(stats.total_examples, 3);
    assert_eq!(stats.successful_examples, 2);
    assert_eq!(stats.failed_examples, 1);
}

#[test]
fn test_training_step() {
    let mut predictor = StrategyPredictor::new();
    let prop = make_theorem("test");

    let example = TrainingExample::from_verification(
        &prop,
        BackendId::Lean4,
        vec!["simp".to_string()],
        1.0,
        true,
    );

    // Training should not panic
    predictor.train(&[example], 0.01, 1);
}

#[test]
fn test_dense_layer_forward() {
    let layer = DenseLayer::new(4, 2);

    let input = vec![1.0, 0.5, -0.5, 0.0];
    let output = layer.forward(&input);

    assert_eq!(output.len(), 2);
}

#[test]
fn test_softmax() {
    let logits = vec![1.0, 2.0, 3.0];
    let probs = softmax(&logits);

    // Sum should be 1.0
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // All probs should be positive
    assert!(probs.iter().all(|&p| p > 0.0));

    // Highest logit should have highest prob
    assert!(probs[2] > probs[1]);
    assert!(probs[1] > probs[0]);
}

#[test]
fn test_normalize() {
    assert_eq!(normalize(0.0, 10.0), 0.0);
    assert!(normalize(10.0, 10.0) > 0.7); // tanh(1.0) ≈ 0.76
    assert!(normalize(100.0, 10.0) > 0.99); // saturates near 1.0
}

#[test]
fn test_backend_idx_roundtrip() {
    for idx in 0..NUM_BACKENDS {
        let backend = idx_to_backend(idx);
        let back_idx = backend_to_idx(backend);
        assert_eq!(idx, back_idx);
    }
}

#[test]
fn test_ml_predictor_with_assistant() {
    let predictor = StrategyPredictor::new();
    let assistant = crate::ProofAssistant::with_ml_predictor(predictor);

    let prop = make_theorem("test");
    let strategy = assistant.recommend_strategy(&prop);

    // With ML predictor, should get ML-based rationale
    assert!(!strategy.rationale.is_empty());
}

#[test]
fn test_from_corpus_entries() {
    let prop1 = make_theorem("corpus_thm1");
    let prop2 = make_temporal("corpus_temp1");
    let tactics1: Vec<String> = vec!["simp".to_string(), "decide".to_string()];
    let tactics2: Vec<String> = vec!["model_check".to_string()];

    let entries: Vec<(&Property, BackendId, &[String], f64)> = vec![
        (&prop1, BackendId::Lean4, &tactics1, 1.5),
        (&prop2, BackendId::TlaPlus, &tactics2, 3.0),
    ];

    let generator = TrainingDataGenerator::from_corpus_entries(entries);
    let stats = generator.stats();

    assert_eq!(stats.total_examples, 2);
    assert_eq!(stats.successful_examples, 2);
    assert_eq!(stats.examples_per_backend.get(&BackendId::Lean4), Some(&1));
    assert_eq!(
        stats.examples_per_backend.get(&BackendId::TlaPlus),
        Some(&1)
    );
}

#[test]
fn test_train_from_corpus() {
    let mut predictor = StrategyPredictor::new();
    let prop1 = make_theorem("train_thm1");
    let prop2 = make_theorem("train_thm2");
    let tactics: Vec<String> = vec!["simp".to_string()];

    let entries: Vec<(&Property, BackendId, &[String], f64)> = vec![
        (&prop1, BackendId::Lean4, &tactics, 1.0),
        (&prop2, BackendId::Lean4, &tactics, 2.0),
    ];

    let stats = predictor.train_from_corpus(entries, 0.01, 5);

    assert_eq!(stats.total_examples, 2);
    assert_eq!(stats.successful_examples, 2);
}

#[test]
fn test_train_and_save() {
    use tempfile::tempdir;

    let mut predictor = StrategyPredictor::new();
    let prop = make_theorem("save_test");
    let tactics: Vec<String> = vec!["simp".to_string()];

    let entries: Vec<(&Property, BackendId, &[String], f64)> =
        vec![(&prop, BackendId::Lean4, &tactics, 1.0)];

    let dir = tempdir().unwrap();
    let model_path = dir.path().join("test_model.json");

    let stats = predictor
        .train_and_save(entries, 0.01, 5, &model_path)
        .unwrap();

    assert_eq!(stats.total_examples, 1);
    assert!(model_path.exists());

    // Verify we can load the saved model
    let loaded = StrategyPredictor::load(&model_path).unwrap();
    let prediction = loaded.predict_backend(&PropertyFeatureVector::from_property(&prop));
    assert!(prediction.confidence >= 0.0);
}

#[test]
fn test_train_with_metrics() {
    let mut predictor = StrategyPredictor::new();

    // Create training data with different backends
    let props: Vec<Property> = (0..10)
        .map(|i| make_theorem(&format!("metrics_thm_{}", i)))
        .collect();

    let examples: Vec<TrainingExample> = props
        .iter()
        .enumerate()
        .map(|(i, p)| {
            TrainingExample::from_verification(
                p,
                if i % 2 == 0 {
                    BackendId::Lean4
                } else {
                    BackendId::TlaPlus
                },
                vec!["simp".to_string()],
                1.0 + i as f64 * 0.5,
                true,
            )
        })
        .collect();

    // Train with validation split
    let history = predictor.train_with_metrics(&examples, 0.01, 5, 0.2);

    // Should have metrics for all epochs
    assert_eq!(history.epochs.len(), 5);

    // Each epoch should have train metrics
    for epoch in &history.epochs {
        assert!(epoch.train_loss >= 0.0);
        assert!(epoch.train_accuracy >= 0.0);
        assert!(epoch.train_accuracy <= 1.0);
    }

    // Should have validation metrics (20% of 10 = 2 examples for validation)
    assert!(history.epochs[0].val_loss.is_some());
    assert!(history.epochs[0].val_accuracy.is_some());

    // History helpers should work
    assert!(history.final_train_accuracy().is_some());
    assert!(history.final_val_accuracy().is_some());
}

#[test]
fn test_train_with_metrics_no_validation() {
    let mut predictor = StrategyPredictor::new();

    let props: Vec<Property> = (0..5)
        .map(|i| make_theorem(&format!("noval_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = props
        .iter()
        .map(|p| {
            TrainingExample::from_verification(
                p,
                BackendId::Lean4,
                vec!["simp".to_string()],
                1.0,
                true,
            )
        })
        .collect();

    // Train with 0.0 validation split (no validation)
    let history = predictor.train_with_metrics(&examples, 0.01, 3, 0.0);

    assert_eq!(history.epochs.len(), 3);
    // No validation metrics
    assert!(history.epochs[0].val_loss.is_none());
    assert!(history.epochs[0].val_accuracy.is_none());
}

#[test]
fn test_train_with_regularization() {
    let mut predictor = StrategyPredictor::new();

    // Create training data with different backends
    let props: Vec<Property> = (0..15)
        .map(|i| make_theorem(&format!("reg_thm_{}", i)))
        .collect();

    let examples: Vec<TrainingExample> = props
        .iter()
        .enumerate()
        .map(|(i, p)| {
            TrainingExample::from_verification(
                p,
                if i % 3 == 0 {
                    BackendId::Lean4
                } else if i % 3 == 1 {
                    BackendId::TlaPlus
                } else {
                    BackendId::Kani
                },
                vec!["simp".to_string()],
                1.0 + i as f64 * 0.5,
                true,
            )
        })
        .collect();

    // Train with weight decay (L2 regularization)
    let history = predictor.train_with_regularization(
        &examples, 0.05,  // learning_rate
        10,    // epochs
        0.2,   // validation_split
        0.001, // weight_decay
    );

    // Should have metrics for all epochs
    assert_eq!(history.epochs.len(), 10);

    // Each epoch should have train metrics
    for epoch in &history.epochs {
        assert!(epoch.train_loss >= 0.0);
        assert!(epoch.train_accuracy >= 0.0);
        assert!(epoch.train_accuracy <= 1.0);
    }

    // Should have validation metrics
    assert!(history.epochs[0].val_loss.is_some());
    assert!(history.epochs[0].val_accuracy.is_some());

    // History helpers should work
    assert!(history.final_train_accuracy().is_some());
    assert!(history.final_val_accuracy().is_some());
}

#[test]
fn test_full_backpropagation_updates_hidden_layers() {
    // Test that full backpropagation updates weights when gradients can flow.
    // We manually set up a layer with known weights to ensure gradient flow.
    let mut layer = DenseLayer::new(4, 3);

    // Set some weights to positive values to ensure positive pre-activations
    layer.weights[0][0] = 0.5;
    layer.weights[1][0] = 0.3;
    layer.weights[2][1] = 0.4;
    layer.weights[3][2] = 0.6;

    // Use positive input
    let input = vec![1.0, 0.5, 0.8, 0.3];

    // Verify we get some positive outputs
    let pre_activation = layer.forward(&input);
    let has_positive = pre_activation.iter().any(|&x| x > 0.0);
    assert!(
        has_positive,
        "Test setup: need at least one active neuron. Pre-activations: {:?}",
        pre_activation
    );

    // Capture initial weight and bias
    let initial_weight = layer.weights[0][0];
    let initial_bias = layer.biases[0];

    // Simulate training: backward pass with a gradient
    let output_grad = vec![0.5, -0.3, 0.2];
    layer.backward(&input, &output_grad, 0.1, 0.0);

    // Weights and biases should have changed
    let final_weight = layer.weights[0][0];
    let final_bias = layer.biases[0];

    assert!(
        (initial_weight - final_weight).abs() > 1e-10,
        "Weights should change after backward pass: {} -> {}",
        initial_weight,
        final_weight
    );
    assert!(
        (initial_bias - final_bias).abs() > 1e-10,
        "Biases should change after backward pass: {} -> {}",
        initial_bias,
        final_bias
    );
}

#[test]
fn test_backpropagation_with_relu_gradient_masking() {
    // Test that ReLU derivative correctly masks gradients
    let pre_activation = vec![0.5, -0.3, 0.1, -0.8, 0.0];
    let incoming_grad = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    let masked_grad = relu_derivative(&pre_activation, &incoming_grad);

    // Only neurons with positive pre-activation should pass gradient
    assert!(
        (masked_grad[0] - 1.0).abs() < 1e-10,
        "Positive pre-activation passes gradient"
    );
    assert!(
        (masked_grad[1] - 0.0).abs() < 1e-10,
        "Negative pre-activation blocks gradient"
    );
    assert!(
        (masked_grad[2] - 1.0).abs() < 1e-10,
        "Positive pre-activation passes gradient"
    );
    assert!(
        (masked_grad[3] - 0.0).abs() < 1e-10,
        "Negative pre-activation blocks gradient"
    );
    assert!(
        (masked_grad[4] - 0.0).abs() < 1e-10,
        "Zero pre-activation blocks gradient"
    );
}

#[test]
fn test_dense_layer_backward() {
    // Test the backward pass of DenseLayer
    let mut layer = DenseLayer::new(4, 3);

    let input = vec![1.0, 0.5, -0.5, 0.0];
    let output_grad = vec![0.1, -0.2, 0.15];

    // Capture initial weights
    let initial_weights: Vec<Vec<f64>> = layer.weights.clone();

    // Run backward pass
    let input_grad = layer.backward(&input, &output_grad, 0.1, 0.0);

    // Input gradient should have same size as input
    assert_eq!(input_grad.len(), 4);

    // Weights should have changed
    let weights_changed =
        layer
            .weights
            .iter()
            .zip(initial_weights.iter())
            .any(|(new_row, old_row)| {
                new_row
                    .iter()
                    .zip(old_row.iter())
                    .any(|(new, old)| (new - old).abs() > 1e-10)
            });
    assert!(weights_changed, "weights should change after backward pass");
}

#[test]
fn test_weight_decay_shrinks_weights() {
    // Test that weight decay actually reduces weight magnitudes over time
    let mut predictor = StrategyPredictor::new();

    // Create training examples
    let props: Vec<Property> = (0..10)
        .map(|i| make_theorem(&format!("wd_thm_{}", i)))
        .collect();

    let examples: Vec<TrainingExample> = props
        .iter()
        .map(|p| {
            TrainingExample::from_verification(
                p,
                BackendId::Lean4,
                vec!["simp".to_string()],
                1.0,
                true,
            )
        })
        .collect();

    // Compute initial L2 norm of weights
    let initial_l2: f64 = predictor
        .hidden1
        .weights
        .iter()
        .flat_map(|row| row.iter())
        .map(|w| w * w)
        .sum::<f64>()
        .sqrt();

    // Train with strong weight decay
    predictor.train_with_regularization(
        &examples, 0.1, // learning_rate
        20,  // epochs
        0.0, // no validation split
        0.1, // strong weight_decay
    );

    // Compute final L2 norm
    let final_l2: f64 = predictor
        .hidden1
        .weights
        .iter()
        .flat_map(|row| row.iter())
        .map(|w| w * w)
        .sum::<f64>()
        .sqrt();

    // With strong weight decay, weights should shrink
    assert!(
        final_l2 < initial_l2,
        "weight decay should reduce weight magnitude: initial={}, final={}",
        initial_l2,
        final_l2
    );
}

#[test]
fn test_evaluate_model() {
    let mut predictor = StrategyPredictor::new();

    // Create and train on some data
    let train_props: Vec<Property> = (0..20)
        .map(|i| make_theorem(&format!("train_{}", i)))
        .collect();
    let train_examples: Vec<TrainingExample> = train_props
        .iter()
        .map(|p| {
            TrainingExample::from_verification(
                p,
                BackendId::Lean4,
                vec!["simp".to_string()],
                1.0,
                true,
            )
        })
        .collect();

    predictor.train(&train_examples, 0.05, 20);

    // Evaluate on test data
    let test_props: Vec<Property> = (0..5)
        .map(|i| make_theorem(&format!("test_{}", i)))
        .collect();
    let test_examples: Vec<TrainingExample> = test_props
        .iter()
        .map(|p| {
            TrainingExample::from_verification(
                p,
                BackendId::Lean4,
                vec!["simp".to_string()],
                1.0,
                true,
            )
        })
        .collect();

    let eval = predictor.evaluate(&test_examples);

    // Basic validation
    assert_eq!(eval.total_examples, 5);
    assert!(eval.accuracy >= 0.0 && eval.accuracy <= 1.0);
    assert!(eval.loss >= 0.0);
    assert!(eval.correct_predictions <= eval.total_examples);
}

#[test]
fn test_evaluate_empty_data() {
    let predictor = StrategyPredictor::new();
    let eval = predictor.evaluate(&[]);

    assert_eq!(eval.total_examples, 0);
    assert_eq!(eval.accuracy, 0.0);
}

#[test]
fn test_evaluation_result_methods() {
    let eval = EvaluationResult {
        accuracy: 0.8,
        loss: 0.5,
        total_examples: 100,
        correct_predictions: 80,
        per_backend_accuracy: vec![(BackendId::Lean4, 0.9), (BackendId::TlaPlus, 0.7)]
            .into_iter()
            .collect(),
        avg_confidence_correct: 0.85,
        avg_confidence_incorrect: 0.45,
    };

    // Test backend_accuracy
    assert_eq!(eval.backend_accuracy(BackendId::Lean4), Some(0.9));
    assert_eq!(eval.backend_accuracy(BackendId::Coq), None);

    // Test is_well_calibrated (correct conf - incorrect conf > 0.1)
    assert!(eval.is_well_calibrated());
}

#[test]
fn test_cross_validation() {
    // Create diverse training data
    let props: Vec<Property> = (0..20)
        .map(|i| {
            if i % 3 == 0 {
                make_temporal(&format!("cv_temp_{}", i))
            } else if i % 3 == 1 {
                make_invariant(&format!("cv_inv_{}", i))
            } else {
                make_theorem(&format!("cv_thm_{}", i))
            }
        })
        .collect();

    let examples: Vec<TrainingExample> = props
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = match i % 4 {
                0 => BackendId::Lean4,
                1 => BackendId::TlaPlus,
                2 => BackendId::Coq,
                _ => BackendId::Kani,
            };
            TrainingExample::from_verification(p, backend, vec!["auto".to_string()], 1.0, true)
        })
        .collect();

    // Run 5-fold cross validation
    let cv_result = StrategyPredictor::cross_validate(&examples, 5, 0.01, 10);

    // Should have 5 folds
    assert_eq!(cv_result.k, 5);
    assert_eq!(cv_result.fold_results.len(), 5);

    // Metrics should be valid
    assert!(cv_result.mean_accuracy >= 0.0 && cv_result.mean_accuracy <= 1.0);
    assert!(cv_result.std_accuracy >= 0.0);
    assert!(cv_result.mean_loss >= 0.0);

    // Confidence interval should work
    let (lower, upper) = cv_result.accuracy_confidence_interval();
    assert!(lower <= cv_result.mean_accuracy);
    assert!(upper >= cv_result.mean_accuracy);
}

#[test]
fn test_cross_validation_edge_cases() {
    // Empty data
    let empty_cv = StrategyPredictor::cross_validate(&[], 5, 0.01, 10);
    assert_eq!(empty_cv.k, 0);

    // k < 2
    let prop = make_theorem("single");
    let examples = vec![TrainingExample::from_verification(
        &prop,
        BackendId::Lean4,
        vec![],
        1.0,
        true,
    )];
    let single_cv = StrategyPredictor::cross_validate(&examples, 1, 0.01, 10);
    assert_eq!(single_cv.k, 0);
}

#[test]
fn test_training_history_helpers() {
    let mut history = TrainingHistory::new();

    // Empty history
    assert!(history.final_train_accuracy().is_none());
    assert!(history.best_val_accuracy().is_none());
    assert!(!history.is_overfitting(3));

    // Add epochs with improving validation
    for i in 0..5 {
        history.epochs.push(EpochMetrics {
            epoch: i,
            train_loss: 1.0 - i as f64 * 0.1,
            train_accuracy: 0.5 + i as f64 * 0.1,
            val_loss: Some(1.2 - i as f64 * 0.1),
            val_accuracy: Some(0.4 + i as f64 * 0.1),
        });
    }

    assert_eq!(history.final_train_accuracy(), Some(0.9));
    assert_eq!(history.final_val_accuracy(), Some(0.8));
    assert_eq!(history.best_val_accuracy(), Some((4, 0.8)));

    // Not overfitting (val loss decreasing)
    assert!(!history.is_overfitting(3));

    // Now add epochs where validation loss increases
    for i in 5..8 {
        history.epochs.push(EpochMetrics {
            epoch: i,
            train_loss: 0.5 - (i - 5) as f64 * 0.1, // Train keeps improving
            train_accuracy: 0.95 + (i - 5) as f64 * 0.01,
            val_loss: Some(0.7 + (i - 5) as f64 * 0.2), // Val gets worse
            val_accuracy: Some(0.8 - (i - 5) as f64 * 0.05),
        });
    }

    // Should detect overfitting
    assert!(history.is_overfitting(3));
}

#[test]
fn test_cv_result_methods() {
    let cv_result = CrossValidationResult {
        k: 5,
        mean_accuracy: 0.75,
        std_accuracy: 0.15,
        mean_loss: 0.8,
        std_loss: 0.1,
        fold_results: vec![],
    };

    // Test confidence interval
    let (lower, upper) = cv_result.accuracy_confidence_interval();
    assert!(lower < cv_result.mean_accuracy);
    assert!(upper > cv_result.mean_accuracy);

    // Test high variance detection
    assert!(cv_result.has_high_variance()); // std > 0.1

    let low_var_result = CrossValidationResult {
        k: 5,
        mean_accuracy: 0.8,
        std_accuracy: 0.05, // Low variance
        ..cv_result
    };
    assert!(!low_var_result.has_high_variance());
}

#[test]
fn test_early_stopping_config_default() {
    let config = EarlyStoppingConfig::default();
    assert_eq!(config.patience, 5);
    assert!((config.min_delta - 0.001).abs() < 1e-9);
    assert!(config.restore_best_weights);
}

#[test]
fn test_early_stopping_config_new() {
    let config = EarlyStoppingConfig::new(10, 0.01);
    assert_eq!(config.patience, 10);
    assert!((config.min_delta - 0.01).abs() < 1e-9);
    assert!(config.restore_best_weights);
}

#[test]
fn test_early_stopping_config_builder() {
    let config = EarlyStoppingConfig::new(3, 0.005).with_restore_best(false);
    assert_eq!(config.patience, 3);
    assert!(!config.restore_best_weights);
}

#[test]
fn test_train_with_early_stopping_empty_data() {
    let mut predictor = StrategyPredictor::new();
    let result =
        predictor.train_with_early_stopping(&[], 0.01, 100, 0.2, EarlyStoppingConfig::default());

    assert!(!result.stopped_early);
    assert_eq!(result.final_epoch, 0);
    assert_eq!(result.history.epochs.len(), 0);
}

#[test]
fn test_train_with_early_stopping_basic() {
    let mut predictor = StrategyPredictor::new();

    // Create training data with multiple backends
    let props: Vec<Property> = (0..30)
        .map(|i| make_theorem(&format!("es_thm_{}", i)))
        .collect();

    let examples: Vec<TrainingExample> = props
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 2 == 0 {
                BackendId::Lean4
            } else {
                BackendId::TlaPlus
            };
            TrainingExample::from_verification(
                p,
                backend,
                vec!["simp".to_string()],
                1.0 + i as f64 * 0.1,
                true,
            )
        })
        .collect();

    let config = EarlyStoppingConfig::new(3, 0.001);
    let result = predictor.train_with_early_stopping(&examples, 0.05, 100, 0.2, config);

    // Should have trained for at least a few epochs
    assert!(result.history.epochs.len() >= 3);

    // Each epoch should have validation metrics
    for epoch in &result.history.epochs {
        assert!(epoch.val_loss.is_some());
        assert!(epoch.val_accuracy.is_some());
    }

    // Best epoch should be valid
    assert!(result.best_epoch < result.history.epochs.len());
    assert!(result.best_val_loss.is_finite());
}

#[test]
fn test_train_with_early_stopping_triggers() {
    let mut predictor = StrategyPredictor::new();

    // Create simple training data that should converge quickly
    let props: Vec<Property> = (0..20)
        .map(|i| make_theorem(&format!("trigger_thm_{}", i)))
        .collect();

    let examples: Vec<TrainingExample> = props
        .iter()
        .map(|p| {
            TrainingExample::from_verification(
                p,
                BackendId::Lean4,
                vec!["simp".to_string()],
                1.0,
                true,
            )
        })
        .collect();

    // Use very low patience to force early stopping
    let config = EarlyStoppingConfig::new(2, 0.0001);
    let result = predictor.train_with_early_stopping(&examples, 0.1, 1000, 0.2, config);

    // With patience=2 and 1000 max epochs, should have stopped early
    // (unless the model keeps improving perfectly which is unlikely)
    assert!(
        result.stopped_early || result.final_epoch == 999,
        "Expected early stop or full training"
    );

    // If stopped early, should have been due to no improvement
    if result.stopped_early {
        assert!(result.epochs_without_improvement >= 2);
    }
}

#[test]
fn test_train_with_early_stopping_restore_weights() {
    let mut predictor = StrategyPredictor::new();

    let props: Vec<Property> = (0..30)
        .map(|i| make_theorem(&format!("restore_thm_{}", i)))
        .collect();

    let examples: Vec<TrainingExample> = props
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = match i % 3 {
                0 => BackendId::Lean4,
                1 => BackendId::TlaPlus,
                _ => BackendId::Coq,
            };
            TrainingExample::from_verification(p, backend, vec![], 1.0, true)
        })
        .collect();

    let config = EarlyStoppingConfig::new(3, 0.001).with_restore_best(true);
    let result = predictor.train_with_early_stopping(&examples, 0.05, 50, 0.2, config);

    // After training, model should be at best weights
    // Test by evaluating - accuracy should be reasonable
    let eval = predictor.evaluate(&examples[..5]);
    assert!(eval.accuracy >= 0.0 && eval.accuracy <= 1.0);

    // Best val loss should be finite and positive
    assert!(result.best_val_loss > 0.0);
    assert!(result.best_val_loss.is_finite());
}

#[test]
fn test_train_with_early_stopping_no_restore_weights() {
    let mut predictor = StrategyPredictor::new();

    let props: Vec<Property> = (0..20)
        .map(|i| make_theorem(&format!("norestore_thm_{}", i)))
        .collect();

    let examples: Vec<TrainingExample> = props
        .iter()
        .map(|p| {
            TrainingExample::from_verification(
                p,
                BackendId::Lean4,
                vec!["simp".to_string()],
                1.0,
                true,
            )
        })
        .collect();

    let config = EarlyStoppingConfig::new(2, 0.001).with_restore_best(false);
    let result = predictor.train_with_early_stopping(&examples, 0.05, 100, 0.2, config);

    // Training should complete without issues
    assert!(!result.history.epochs.is_empty());
    assert!(result.best_val_loss.is_finite());
}

#[test]
fn test_early_stopping_result_converged() {
    // Test converged() method
    let result_converged = EarlyStoppingResult {
        history: TrainingHistory::new(),
        stopped_early: true,
        final_epoch: 10,
        best_epoch: 8,
        best_val_loss: 0.5, // < 1.0
        epochs_without_improvement: 3,
    };
    assert!(result_converged.converged());

    let result_not_converged = EarlyStoppingResult {
        history: TrainingHistory::new(),
        stopped_early: true,
        final_epoch: 10,
        best_epoch: 8,
        best_val_loss: 1.5, // > 1.0
        epochs_without_improvement: 3,
    };
    assert!(!result_not_converged.converged());

    let result_not_stopped = EarlyStoppingResult {
        history: TrainingHistory::new(),
        stopped_early: false, // Did not stop early
        final_epoch: 100,
        best_epoch: 95,
        best_val_loss: 0.3,
        epochs_without_improvement: 0,
    };
    assert!(!result_not_stopped.converged());
}

#[test]
fn test_early_stopping_result_improvement_ratio() {
    let mut history = TrainingHistory::new();
    history.epochs.push(EpochMetrics {
        epoch: 0,
        train_loss: 2.0,
        train_accuracy: 0.3,
        val_loss: Some(2.0),
        val_accuracy: Some(0.3),
    });
    history.epochs.push(EpochMetrics {
        epoch: 1,
        train_loss: 1.0,
        train_accuracy: 0.6,
        val_loss: Some(1.0),
        val_accuracy: Some(0.6),
    });

    let result = EarlyStoppingResult {
        history,
        stopped_early: true,
        final_epoch: 1,
        best_epoch: 1,
        best_val_loss: 1.0,
        epochs_without_improvement: 2,
    };

    // improvement_ratio = best_val_loss / first_val_loss = 1.0 / 2.0 = 0.5
    let ratio = result.improvement_ratio().unwrap();
    assert!((ratio - 0.5).abs() < 1e-9);

    // Test with empty history
    let empty_result = EarlyStoppingResult {
        history: TrainingHistory::new(),
        stopped_early: false,
        final_epoch: 0,
        best_epoch: 0,
        best_val_loss: f64::INFINITY,
        epochs_without_improvement: 0,
    };
    assert!(empty_result.improvement_ratio().is_none());
}

#[test]
fn test_train_with_early_stopping_small_dataset() {
    let mut predictor = StrategyPredictor::new();

    // Very small dataset - edge case
    let props: Vec<Property> = (0..5)
        .map(|i| make_theorem(&format!("small_thm_{}", i)))
        .collect();

    let examples: Vec<TrainingExample> = props
        .iter()
        .map(|p| {
            TrainingExample::from_verification(
                p,
                BackendId::Lean4,
                vec!["simp".to_string()],
                1.0,
                true,
            )
        })
        .collect();

    let config = EarlyStoppingConfig::default();
    let result = predictor.train_with_early_stopping(&examples, 0.01, 10, 0.2, config);

    // Should handle small dataset gracefully
    // 5 examples with 0.2 split = 1 val, 4 train
    assert!(result.history.epochs.len() <= 10);
}

// Learning rate scheduler tests

#[test]
fn test_lr_scheduler_constant() {
    let scheduler = LearningRateScheduler::Constant;
    assert_eq!(scheduler.name(), "constant");

    let initial_lr = 0.01;
    for epoch in 0..100 {
        assert_eq!(scheduler.get_lr(initial_lr, epoch, &[]), initial_lr);
    }
}

#[test]
fn test_lr_scheduler_step_decay() {
    let scheduler = LearningRateScheduler::step(10, 0.5);
    assert_eq!(scheduler.name(), "step");

    let initial_lr = 0.1;

    // Epochs 0-9: no reduction
    for epoch in 0..10 {
        let lr = scheduler.get_lr(initial_lr, epoch, &[]);
        assert!((lr - 0.1).abs() < 1e-9, "epoch {}: lr={}", epoch, lr);
    }

    // Epochs 10-19: one reduction (0.1 * 0.5 = 0.05)
    for epoch in 10..20 {
        let lr = scheduler.get_lr(initial_lr, epoch, &[]);
        assert!((lr - 0.05).abs() < 1e-9, "epoch {}: lr={}", epoch, lr);
    }

    // Epochs 20-29: two reductions (0.1 * 0.5^2 = 0.025)
    for epoch in 20..30 {
        let lr = scheduler.get_lr(initial_lr, epoch, &[]);
        assert!((lr - 0.025).abs() < 1e-9, "epoch {}: lr={}", epoch, lr);
    }
}

#[test]
fn test_lr_scheduler_exponential_decay() {
    let scheduler = LearningRateScheduler::exponential(0.9);
    assert_eq!(scheduler.name(), "exponential");

    let initial_lr = 0.1;

    // Epoch 0: 0.1 * 0.9^0 = 0.1
    let lr_0 = scheduler.get_lr(initial_lr, 0, &[]);
    assert!((lr_0 - 0.1).abs() < 1e-9);

    // Epoch 1: 0.1 * 0.9^1 = 0.09
    let lr_1 = scheduler.get_lr(initial_lr, 1, &[]);
    assert!((lr_1 - 0.09).abs() < 1e-9);

    // Epoch 10: 0.1 * 0.9^10 ≈ 0.0348678
    let lr_10 = scheduler.get_lr(initial_lr, 10, &[]);
    assert!((lr_10 - 0.0348678).abs() < 1e-5);
}

#[test]
fn test_lr_scheduler_cosine_annealing() {
    let scheduler = LearningRateScheduler::cosine(0.001, 10);
    assert_eq!(scheduler.name(), "cosine");

    let initial_lr = 0.1;

    // Epoch 0: maximum (cos(0) = 1, so lr = min + (max-min)*1 = max)
    let lr_0 = scheduler.get_lr(initial_lr, 0, &[]);
    assert!((lr_0 - 0.1).abs() < 1e-9);

    // Epoch 5: midpoint (cos(π/2) = 0, so lr is between min and max)
    // lr = 0.001 + (0.1 - 0.001) * (1 + 0) / 2 = 0.0505
    let lr_5 = scheduler.get_lr(initial_lr, 5, &[]);
    assert!((lr_5 - 0.0505).abs() < 1e-6);

    // Epoch 10: back to maximum (cycle restarts)
    let lr_10 = scheduler.get_lr(initial_lr, 10, &[]);
    assert!((lr_10 - 0.1).abs() < 1e-9);

    // Near the end of cycle (epoch 9): approaching minimum
    let lr_9 = scheduler.get_lr(initial_lr, 9, &[]);
    assert!(lr_9 < 0.01); // Should be close to min
}

#[test]
fn test_lr_scheduler_reduce_on_plateau() {
    let scheduler = LearningRateScheduler::reduce_on_plateau(0.5, 3);
    assert_eq!(scheduler.name(), "reduce_on_plateau");

    let initial_lr = 0.1;

    // No history yet - use initial lr
    assert_eq!(scheduler.get_lr(initial_lr, 0, &[]), initial_lr);

    // Not enough history for patience
    let short_history = vec![1.0, 0.9, 0.8];
    assert_eq!(scheduler.get_lr(initial_lr, 3, &short_history), initial_lr);

    // Improving history - no reduction
    let improving_history = vec![2.0, 1.5, 1.0, 0.5, 0.4, 0.3];
    let lr = scheduler.get_lr(initial_lr, 6, &improving_history);
    assert!((lr - initial_lr).abs() < 1e-9);

    // Stagnant history - should trigger reduction
    let stagnant_history = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let lr = scheduler.get_lr(initial_lr, 5, &stagnant_history);
    assert!((lr - 0.05).abs() < 1e-9); // 0.1 * 0.5 = 0.05
}

#[test]
fn test_lr_scheduler_warmup_decay() {
    let scheduler = LearningRateScheduler::warmup_decay(5, 0.1, 0.9);
    assert_eq!(scheduler.name(), "warmup_decay");

    let initial_lr = 0.01;

    // During warmup (linear increase from 0.01 to 0.1)
    let lr_0 = scheduler.get_lr(initial_lr, 0, &[]);
    // lr = 0.01 + (0.1 - 0.01) * (0+1)/5 = 0.01 + 0.018 = 0.028
    assert!((lr_0 - 0.028).abs() < 1e-9);

    // At warmup end (epoch 4)
    let lr_4 = scheduler.get_lr(initial_lr, 4, &[]);
    // lr = 0.01 + (0.1 - 0.01) * 5/5 = 0.1
    assert!((lr_4 - 0.1).abs() < 1e-9);

    // After warmup (exponential decay from peak)
    let lr_5 = scheduler.get_lr(initial_lr, 5, &[]);
    // lr = 0.1 * 0.9^0 = 0.1
    assert!((lr_5 - 0.1).abs() < 1e-9);

    let lr_6 = scheduler.get_lr(initial_lr, 6, &[]);
    // lr = 0.1 * 0.9^1 = 0.09
    assert!((lr_6 - 0.09).abs() < 1e-9);
}

#[test]
fn test_train_with_scheduler_step() {
    let mut predictor = StrategyPredictor::new();

    let props: Vec<Property> = (0..20)
        .map(|i| make_theorem(&format!("sched_thm_{}", i)))
        .collect();

    let examples: Vec<TrainingExample> = props
        .iter()
        .map(|p| {
            TrainingExample::from_verification(
                p,
                BackendId::Lean4,
                vec!["simp".to_string()],
                1.0,
                true,
            )
        })
        .collect();

    let scheduler = LearningRateScheduler::step(5, 0.5);
    let result = predictor.train_with_scheduler(&examples, 0.1, 15, 0.2, scheduler);

    // Should have 15 epochs
    assert_eq!(result.history.epochs.len(), 15);
    assert_eq!(result.lr_history.len(), 15);

    // Check learning rate history
    assert!((result.lr_history[0] - 0.1).abs() < 1e-9);
    assert!((result.lr_history[5] - 0.05).abs() < 1e-9); // After first step
    assert!((result.lr_history[10] - 0.025).abs() < 1e-9); // After second step

    // Check scheduler name
    assert_eq!(result.scheduler_name, "step");
    assert!(!result.used_early_stopping);
    assert!(result.early_stopping_info.is_none());
}

#[test]
fn test_train_with_scheduler_exponential() {
    let mut predictor = StrategyPredictor::new();

    let props: Vec<Property> = (0..15)
        .map(|i| make_theorem(&format!("exp_thm_{}", i)))
        .collect();

    let examples: Vec<TrainingExample> = props
        .iter()
        .map(|p| {
            TrainingExample::from_verification(
                p,
                BackendId::TlaPlus,
                vec!["model_check".to_string()],
                2.0,
                true,
            )
        })
        .collect();

    let scheduler = LearningRateScheduler::exponential(0.95);
    let result = predictor.train_with_scheduler(&examples, 0.1, 10, 0.2, scheduler);

    assert_eq!(result.lr_history.len(), 10);
    assert_eq!(result.scheduler_name, "exponential");

    // Learning rate should decrease monotonically
    for i in 1..result.lr_history.len() {
        assert!(result.lr_history[i] < result.lr_history[i - 1]);
    }

    // Final lr should be approximately 0.1 * 0.95^9
    let expected_final = 0.1 * 0.95_f64.powi(9);
    assert!((result.final_lr - expected_final).abs() < 1e-9);
}

#[test]
fn test_train_with_scheduler_cosine() {
    let mut predictor = StrategyPredictor::new();

    let props: Vec<Property> = (0..20)
        .map(|i| make_theorem(&format!("cos_thm_{}", i)))
        .collect();

    let examples: Vec<TrainingExample> = props
        .iter()
        .map(|p| {
            TrainingExample::from_verification(
                p,
                BackendId::Kani,
                vec!["symbolic_exec".to_string()],
                1.5,
                true,
            )
        })
        .collect();

    let scheduler = LearningRateScheduler::cosine(0.001, 10);
    let result = predictor.train_with_scheduler(&examples, 0.1, 10, 0.2, scheduler);

    assert_eq!(result.scheduler_name, "cosine");

    // First lr should be max (0.1)
    assert!((result.lr_history[0] - 0.1).abs() < 1e-9);

    // Middle lr should be at midpoint (~0.0505)
    // lr = min + (max - min) * (1 + cos(π*5/10)) / 2 = 0.001 + 0.099 * 0.5 = 0.0505
    assert!((result.lr_history[5] - 0.0505).abs() < 1e-6);
}

#[test]
fn test_train_with_scheduler_and_early_stopping() {
    let mut predictor = StrategyPredictor::new();

    let props: Vec<Property> = (0..30)
        .map(|i| make_theorem(&format!("full_thm_{}", i)))
        .collect();

    let examples: Vec<TrainingExample> = props
        .iter()
        .map(|p| {
            TrainingExample::from_verification(
                p,
                BackendId::Coq,
                vec!["auto".to_string()],
                0.8,
                true,
            )
        })
        .collect();

    let scheduler = LearningRateScheduler::step(10, 0.5);
    let early_stopping = EarlyStoppingConfig::new(5, 0.001);
    let result = predictor.train_with_scheduler_and_early_stopping(
        &examples,
        0.1,
        100,
        0.2,
        scheduler,
        early_stopping,
    );

    assert_eq!(result.scheduler_name, "step");
    assert!(result.used_early_stopping);
    assert!(result.early_stopping_info.is_some());

    let es_info = result.early_stopping_info.as_ref().unwrap();
    // Training should have run for at least some epochs
    assert!(!result.history.epochs.is_empty());

    // LR history should match epoch count
    assert_eq!(result.lr_history.len(), result.history.epochs.len());

    // Best val loss should be tracked
    assert!(es_info.best_val_loss.is_finite());
}

#[test]
fn test_train_with_scheduler_empty_data() {
    let mut predictor = StrategyPredictor::new();
    let examples: Vec<TrainingExample> = vec![];

    let scheduler = LearningRateScheduler::exponential(0.9);
    let result = predictor.train_with_scheduler(&examples, 0.1, 10, 0.2, scheduler);

    assert!(result.history.epochs.is_empty());
    assert!(result.lr_history.is_empty());
    assert_eq!(result.final_lr, 0.1);
}

#[test]
fn test_scheduled_training_result_helpers() {
    let mut history = TrainingHistory::new();
    history.epochs.push(EpochMetrics {
        epoch: 0,
        train_loss: 2.0,
        train_accuracy: 0.3,
        val_loss: Some(2.0),
        val_accuracy: Some(0.3),
    });
    history.epochs.push(EpochMetrics {
        epoch: 1,
        train_loss: 1.0,
        train_accuracy: 0.6,
        val_loss: Some(1.0),
        val_accuracy: Some(0.6),
    });

    let result = ScheduledTrainingResult {
        history,
        lr_history: vec![0.1, 0.05],
        final_lr: 0.05,
        scheduler_name: "step".to_string(),
        used_early_stopping: false,
        early_stopping_info: None,
    };

    // Test helper methods
    assert_eq!(result.final_accuracy(), Some(0.6));
    assert_eq!(result.final_val_accuracy(), Some(0.6));
    assert_eq!(result.lr_at_epoch(0), Some(0.1));
    assert_eq!(result.lr_at_epoch(1), Some(0.05));
    assert_eq!(result.lr_at_epoch(2), None);

    // lr_reduction_factor = 0.05 / 0.1 = 0.5
    let factor = result.lr_reduction_factor();
    assert!((factor - 0.5).abs() < 1e-9);
}

#[test]
fn test_lr_scheduler_default() {
    let scheduler = LearningRateScheduler::default();
    assert_eq!(scheduler.name(), "constant");

    // Default should be constant
    let lr = scheduler.get_lr(0.1, 100, &[]);
    assert_eq!(lr, 0.1);
}

#[test]
fn test_lr_scheduler_constructor_helpers() {
    // Test all constructor helpers
    let step = LearningRateScheduler::step(10, 0.5);
    assert_eq!(step.name(), "step");

    let exp = LearningRateScheduler::exponential(0.9);
    assert_eq!(exp.name(), "exponential");

    let cos = LearningRateScheduler::cosine(0.001, 20);
    assert_eq!(cos.name(), "cosine");

    let plateau = LearningRateScheduler::reduce_on_plateau(0.1, 5);
    assert_eq!(plateau.name(), "reduce_on_plateau");

    let warmup = LearningRateScheduler::warmup_decay(10, 0.1, 0.95);
    assert_eq!(warmup.name(), "warmup_decay");
}

// ========================================================================
// Hyperparameter Search Tests
// ========================================================================

#[test]
fn test_hyperparameters_default() {
    let hp = Hyperparameters::default();
    assert_eq!(hp.learning_rate, 0.01);
    assert_eq!(hp.epochs, 100);
    assert!((hp.validation_split - 0.2).abs() < 1e-9);
    assert!(matches!(hp.lr_scheduler, LearningRateScheduler::Constant));
    assert!(hp.early_stopping.is_some());
}

#[test]
fn test_hyperparameters_builder() {
    let hp = Hyperparameters::new(0.05, 50)
        .with_validation_split(0.3)
        .with_scheduler(LearningRateScheduler::step(10, 0.5))
        .with_early_stopping(EarlyStoppingConfig::new(3, 0.01));

    assert_eq!(hp.learning_rate, 0.05);
    assert_eq!(hp.epochs, 50);
    assert!((hp.validation_split - 0.3).abs() < 1e-9);
    assert!(matches!(
        hp.lr_scheduler,
        LearningRateScheduler::Step { .. }
    ));
    assert!(hp.early_stopping.is_some());
    assert_eq!(hp.early_stopping.as_ref().unwrap().patience, 3);
}

#[test]
fn test_hyperparameters_without_early_stopping() {
    let hp = Hyperparameters::new(0.01, 100).without_early_stopping();
    assert!(hp.early_stopping.is_none());
}

#[test]
fn test_hyperparameters_validation_split_clamped() {
    let hp = Hyperparameters::new(0.01, 50).with_validation_split(0.8);
    assert!((hp.validation_split - 0.5).abs() < 1e-9); // Clamped to max 0.5

    let hp2 = Hyperparameters::new(0.01, 50).with_validation_split(-0.1);
    assert!((hp2.validation_split - 0.0).abs() < 1e-9); // Clamped to min 0.0
}

#[test]
fn test_grid_search_space_default() {
    let space = GridSearchSpace::default();
    assert_eq!(space.learning_rates.len(), 4);
    assert_eq!(space.epochs.len(), 2);
    assert_eq!(space.validation_splits.len(), 1);
    assert_eq!(space.schedulers.len(), 1);
    assert_eq!(space.patience_values.len(), 1);
}

#[test]
fn test_grid_search_space_builder() {
    let space = GridSearchSpace::new(vec![0.01, 0.1])
        .with_epochs(vec![10, 20, 30])
        .with_validation_splits(vec![0.1, 0.2])
        .with_schedulers(vec![
            LearningRateScheduler::Constant,
            LearningRateScheduler::step(5, 0.5),
        ])
        .with_patience_values(vec![Some(3), None]);

    assert_eq!(space.learning_rates.len(), 2);
    assert_eq!(space.epochs.len(), 3);
    assert_eq!(space.validation_splits.len(), 2);
    assert_eq!(space.schedulers.len(), 2);
    assert_eq!(space.patience_values.len(), 2);
}

#[test]
fn test_grid_search_space_total_configurations() {
    let space = GridSearchSpace::new(vec![0.01, 0.1])
        .with_epochs(vec![10, 20])
        .with_validation_splits(vec![0.2])
        .with_schedulers(vec![LearningRateScheduler::Constant])
        .with_patience_values(vec![Some(5)]);

    // 2 * 2 * 1 * 1 * 1 = 4
    assert_eq!(space.total_configurations(), 4);
}

#[test]
fn test_grid_search_space_generate_configs() {
    let space = GridSearchSpace::new(vec![0.01, 0.05])
        .with_epochs(vec![10])
        .with_validation_splits(vec![0.2])
        .with_schedulers(vec![LearningRateScheduler::Constant])
        .with_patience_values(vec![Some(5)]);

    let configs = space.generate_configs();
    assert_eq!(configs.len(), 2);

    assert!((configs[0].learning_rate - 0.01).abs() < 1e-9);
    assert!((configs[1].learning_rate - 0.05).abs() < 1e-9);
}

#[test]
fn test_random_search_config_default() {
    let config = RandomSearchConfig::default();
    assert_eq!(config.n_iterations, 20);
    assert_eq!(config.seed, 42);
    assert!((config.early_stopping_prob - 0.8).abs() < 1e-9);
}

#[test]
fn test_random_search_config_builder() {
    let config = RandomSearchConfig::new(10)
        .with_lr_range(0.001, 0.1)
        .with_epoch_range(20, 50)
        .with_val_split_range(0.15, 0.25)
        .with_seed(123);

    assert_eq!(config.n_iterations, 10);
    assert_eq!(config.lr_range, (0.001, 0.1));
    assert_eq!(config.epoch_range, (20, 50));
    assert_eq!(config.seed, 123);
}

#[test]
fn test_random_search_config_generate_configs() {
    let config = RandomSearchConfig::new(5).with_seed(42);
    let configs = config.generate_configs();

    assert_eq!(configs.len(), 5);

    // All configs should have valid values
    for hp in &configs {
        assert!(hp.learning_rate > 0.0);
        assert!(hp.epochs > 0);
        assert!(hp.validation_split >= 0.0 && hp.validation_split <= 1.0);
    }
}

#[test]
fn test_random_search_config_reproducibility() {
    let config1 = RandomSearchConfig::new(5).with_seed(42);
    let config2 = RandomSearchConfig::new(5).with_seed(42);

    let configs1 = config1.generate_configs();
    let configs2 = config2.generate_configs();

    // Same seed should produce same configs
    for (hp1, hp2) in configs1.iter().zip(configs2.iter()) {
        assert!((hp1.learning_rate - hp2.learning_rate).abs() < 1e-9);
        assert_eq!(hp1.epochs, hp2.epochs);
    }
}

#[test]
fn test_hyperparameter_result_is_better_than() {
    let better = HyperparameterResult {
        hyperparameters: Hyperparameters::default(),
        val_loss: 0.5,
        val_accuracy: 0.8,
        train_loss: 0.4,
        train_accuracy: 0.85,
        epochs_trained: 50,
        stopped_early: false,
    };

    let worse = HyperparameterResult {
        hyperparameters: Hyperparameters::default(),
        val_loss: 0.7,
        val_accuracy: 0.7,
        train_loss: 0.5,
        train_accuracy: 0.75,
        epochs_trained: 100,
        stopped_early: false,
    };

    assert!(better.is_better_than(&worse));
    assert!(!worse.is_better_than(&better));
}

#[test]
fn test_hyperparameter_result_is_better_than_tie_breaker() {
    // Same val_loss, better accuracy wins
    let better = HyperparameterResult {
        hyperparameters: Hyperparameters::default(),
        val_loss: 0.5,
        val_accuracy: 0.85,
        train_loss: 0.4,
        train_accuracy: 0.85,
        epochs_trained: 50,
        stopped_early: false,
    };

    let worse = HyperparameterResult {
        hyperparameters: Hyperparameters::default(),
        val_loss: 0.5,
        val_accuracy: 0.75,
        train_loss: 0.5,
        train_accuracy: 0.75,
        epochs_trained: 100,
        stopped_early: false,
    };

    assert!(better.is_better_than(&worse));
}

#[test]
fn test_hyperparameter_search_result_helpers() {
    let results = vec![
        HyperparameterResult {
            hyperparameters: Hyperparameters::new(0.01, 50),
            val_loss: 0.8,
            val_accuracy: 0.6,
            train_loss: 0.6,
            train_accuracy: 0.7,
            epochs_trained: 50,
            stopped_early: false,
        },
        HyperparameterResult {
            hyperparameters: Hyperparameters::new(0.05, 100),
            val_loss: 0.5,
            val_accuracy: 0.8,
            train_loss: 0.4,
            train_accuracy: 0.85,
            epochs_trained: 80,
            stopped_early: true,
        },
    ];

    let search_result = HyperparameterSearchResult {
        results,
        best_idx: 1,
        total_evaluated: 2,
        search_method: "test".to_string(),
    };

    assert!((search_result.best_val_loss() - 0.5).abs() < 1e-9);
    assert!((search_result.best_val_accuracy() - 0.8).abs() < 1e-9);
    assert_eq!(search_result.best_hyperparameters().epochs, 100);

    let sorted = search_result.sorted_by_loss();
    assert_eq!(sorted.len(), 2);
    assert!((sorted[0].val_loss - 0.5).abs() < 1e-9); // Best first
    assert!((sorted[1].val_loss - 0.8).abs() < 1e-9);
}

#[test]
fn test_grid_search_basic() {
    // Create minimal training data
    let properties: Vec<_> = (0..20)
        .map(|i| make_theorem(&format!("gs_thm_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 2 == 0 {
                BackendId::Lean4
            } else {
                BackendId::TlaPlus
            };
            TrainingExample::from_verification(p, backend, vec!["simp".to_string()], 1.0, true)
        })
        .collect();

    // Very small search space for fast test
    let space = GridSearchSpace::new(vec![0.05])
        .with_epochs(vec![5])
        .with_patience_values(vec![Some(2)]);

    let result = StrategyPredictor::grid_search(&examples, &space);

    assert_eq!(result.total_evaluated, 1);
    assert_eq!(result.search_method, "grid");
    assert_eq!(result.best_idx, 0);
}

#[test]
fn test_grid_search_finds_best() {
    // Create training data
    let properties: Vec<_> = (0..30)
        .map(|i| make_theorem(&format!("gs2_thm_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 3 == 0 {
                BackendId::Lean4
            } else if i % 3 == 1 {
                BackendId::TlaPlus
            } else {
                BackendId::Kani
            };
            TrainingExample::from_verification(p, backend, vec!["auto".to_string()], 1.0, true)
        })
        .collect();

    let space = GridSearchSpace::new(vec![0.01, 0.1])
        .with_epochs(vec![5])
        .with_patience_values(vec![Some(2)]);

    let result = StrategyPredictor::grid_search(&examples, &space);

    assert_eq!(result.total_evaluated, 2);
    assert!(result.best_idx < 2);

    // Best should have minimum val_loss
    let best_loss = result.best_val_loss();
    for r in &result.results {
        assert!(r.val_loss >= best_loss - 1e-9);
    }
}

#[test]
fn test_random_search_basic() {
    let properties: Vec<_> = (0..20)
        .map(|i| make_theorem(&format!("rs_thm_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 2 == 0 {
                BackendId::Lean4
            } else {
                BackendId::TlaPlus
            };
            TrainingExample::from_verification(p, backend, vec!["simp".to_string()], 1.0, true)
        })
        .collect();

    // Very small random search for fast test
    let config = RandomSearchConfig::new(2)
        .with_epoch_range(3, 5)
        .with_seed(42);

    let result = StrategyPredictor::random_search(&examples, &config);

    assert_eq!(result.total_evaluated, 2);
    assert_eq!(result.search_method, "random");
}

#[test]
fn test_evaluate_hyperparameters_basic_training() {
    let properties: Vec<_> = (0..15)
        .map(|i| make_theorem(&format!("eval_thm_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 2 == 0 {
                BackendId::Lean4
            } else {
                BackendId::TlaPlus
            };
            TrainingExample::from_verification(p, backend, vec!["simp".to_string()], 1.0, true)
        })
        .collect();

    let hp = Hyperparameters::new(0.05, 5).without_early_stopping();
    let result = StrategyPredictor::evaluate_hyperparameters(&examples, hp);

    assert_eq!(result.epochs_trained, 5);
    assert!(!result.stopped_early);
    assert!(result.val_loss.is_finite());
    assert!(result.train_loss.is_finite());
}

#[test]
fn test_evaluate_hyperparameters_with_early_stopping() {
    let properties: Vec<_> = (0..20)
        .map(|i| make_theorem(&format!("es_hp_thm_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 2 == 0 {
                BackendId::Lean4
            } else {
                BackendId::TlaPlus
            };
            TrainingExample::from_verification(p, backend, vec!["simp".to_string()], 1.0, true)
        })
        .collect();

    let hp =
        Hyperparameters::new(0.05, 100).with_early_stopping(EarlyStoppingConfig::new(2, 0.001));
    let result = StrategyPredictor::evaluate_hyperparameters(&examples, hp);

    // May or may not have stopped early, but should have valid results
    assert!(result.epochs_trained <= 100);
    assert!(result.val_loss.is_finite());
}

#[test]
fn test_evaluate_hyperparameters_with_scheduler() {
    let properties: Vec<_> = (0..15)
        .map(|i| make_theorem(&format!("sched_hp_thm_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 2 == 0 {
                BackendId::Lean4
            } else {
                BackendId::TlaPlus
            };
            TrainingExample::from_verification(p, backend, vec!["simp".to_string()], 1.0, true)
        })
        .collect();

    let hp = Hyperparameters::new(0.1, 10)
        .with_scheduler(LearningRateScheduler::step(5, 0.5))
        .without_early_stopping();
    let result = StrategyPredictor::evaluate_hyperparameters(&examples, hp);

    assert_eq!(result.epochs_trained, 10);
    assert!(result.val_loss.is_finite());
}

#[test]
fn test_evaluate_hyperparameters_with_scheduler_and_early_stopping() {
    let properties: Vec<_> = (0..25)
        .map(|i| make_theorem(&format!("combo_hp_thm_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 3 == 0 {
                BackendId::Lean4
            } else if i % 3 == 1 {
                BackendId::TlaPlus
            } else {
                BackendId::Coq
            };
            TrainingExample::from_verification(p, backend, vec!["auto".to_string()], 1.0, true)
        })
        .collect();

    let hp = Hyperparameters::new(0.1, 50)
        .with_scheduler(LearningRateScheduler::exponential(0.95))
        .with_early_stopping(EarlyStoppingConfig::new(3, 0.001));
    let result = StrategyPredictor::evaluate_hyperparameters(&examples, hp);

    assert!(result.epochs_trained <= 50);
    assert!(result.val_loss.is_finite());
}

#[test]
fn test_train_with_best_hyperparameters() {
    let properties: Vec<_> = (0..20)
        .map(|i| make_theorem(&format!("best_hp_thm_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 2 == 0 {
                BackendId::Lean4
            } else {
                BackendId::TlaPlus
            };
            TrainingExample::from_verification(p, backend, vec!["simp".to_string()], 1.0, true)
        })
        .collect();

    let space = GridSearchSpace::new(vec![0.01, 0.05])
        .with_epochs(vec![5])
        .with_patience_values(vec![Some(2)]);

    let search_result = StrategyPredictor::grid_search(&examples, &space);

    let mut predictor = StrategyPredictor::new();
    let result = predictor.train_with_best_hyperparameters(&examples, &search_result);

    // Result should match best hyperparameters
    assert_eq!(
        result.hyperparameters.epochs,
        search_result.best_hyperparameters().epochs
    );
}

#[test]
fn test_simple_rng() {
    let mut rng = SimpleRng::new(42);

    // Values should be in [0, 1)
    for _ in 0..100 {
        let val = rng.next_f64();
        assert!((0.0..1.0).contains(&val));
    }
}

#[test]
fn test_simple_rng_deterministic() {
    let mut rng1 = SimpleRng::new(42);
    let mut rng2 = SimpleRng::new(42);

    for _ in 0..10 {
        assert_eq!(rng1.next_u64(), rng2.next_u64());
    }
}

#[test]
fn test_simple_rng_zero_seed_handled() {
    // Zero seed should be converted to non-zero
    let mut rng = SimpleRng::new(0);
    let val = rng.next_u64();
    assert_ne!(val, 0); // Should produce non-zero values
}

// ========================================================================
// Checkpointing Tests
// ========================================================================

#[test]
fn test_checkpoint_config_defaults() {
    let config = CheckpointConfig::default();
    assert_eq!(config.save_every_n_epochs, 0);
    assert_eq!(config.keep_best_n, 3);
    assert!(config.save_on_improvement);
    assert!(config.include_history);
}

#[test]
fn test_checkpoint_config_builder() {
    let config = CheckpointConfig::new("/tmp/checkpoints")
        .with_save_interval(5)
        .with_keep_best(2)
        .with_save_on_improvement(false)
        .with_history(false);

    assert_eq!(
        config.checkpoint_dir,
        std::path::PathBuf::from("/tmp/checkpoints")
    );
    assert_eq!(config.save_every_n_epochs, 5);
    assert_eq!(config.keep_best_n, 2);
    assert!(!config.save_on_improvement);
    assert!(!config.include_history);
}

#[test]
fn test_checkpoint_filename() {
    let checkpoint = Checkpoint {
        epoch: 5,
        val_loss: 0.123456,
        val_accuracy: 0.85,
        train_loss: 0.1,
        train_accuracy: 0.9,
        model: StrategyPredictor::new(),
        history: None,
        timestamp: "12345".to_string(),
    };

    let filename = checkpoint.filename();
    assert!(filename.contains("epoch_0005"));
    assert!(filename.contains("val_loss"));
    assert!(filename.ends_with(".json"));
}

#[test]
fn test_train_with_checkpointing() {
    let properties: Vec<_> = (0..30)
        .map(|i| make_theorem(&format!("ckpt_thm_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 3 == 0 {
                BackendId::Lean4
            } else if i % 3 == 1 {
                BackendId::TlaPlus
            } else {
                BackendId::Coq
            };
            TrainingExample::from_verification(p, backend, vec!["auto".to_string()], 1.0, true)
        })
        .collect();

    // Use unique temp dir to avoid race conditions in parallel tests
    let unique_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!("dashprove_ckpt_test_{}", unique_id));
    let _ = std::fs::remove_dir_all(&temp_dir);

    let config = CheckpointConfig::new(&temp_dir)
        .with_save_interval(3)
        .with_keep_best(2);

    let mut predictor = StrategyPredictor::new();
    let result = predictor.train_with_checkpointing(&examples, 0.1, 10, 0.2, &config);

    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.total_epochs, 10);
    assert!(result.best_val_loss.is_finite());
    assert!(result.best_epoch < 10);

    // Clean up
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_compute_loss() {
    let predictor = StrategyPredictor::new();
    let prop = make_theorem("loss_test");
    let example = TrainingExample::from_verification(
        &prop,
        BackendId::Lean4,
        vec!["simp".to_string()],
        1.0,
        true,
    );

    let loss = predictor.compute_loss(&example);
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

// ========================================================================
// Cross-Validation Hyperparameter Search Tests
// ========================================================================

#[test]
fn test_cv_hyperparameter_result() {
    let properties: Vec<_> = (0..20)
        .map(|i| make_theorem(&format!("cv_hp_thm_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 2 == 0 {
                BackendId::Lean4
            } else {
                BackendId::TlaPlus
            };
            TrainingExample::from_verification(p, backend, vec!["auto".to_string()], 1.0, true)
        })
        .collect();

    let hp = Hyperparameters::new(0.1, 5);
    let result = StrategyPredictor::evaluate_hyperparameters_cv(&examples, hp, 3);

    assert_eq!(result.k_folds, 3);
    assert_eq!(result.fold_results.len(), 3);
    assert!(result.mean_val_loss.is_finite());
    assert!(result.std_val_loss >= 0.0);
    assert!(result.mean_val_accuracy >= 0.0);
    assert!(result.mean_val_accuracy <= 1.0);
}

#[test]
fn test_grid_search_with_cv() {
    let properties: Vec<_> = (0..20)
        .map(|i| make_theorem(&format!("grid_cv_thm_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 2 == 0 {
                BackendId::Lean4
            } else {
                BackendId::Coq
            };
            TrainingExample::from_verification(p, backend, vec!["simp".to_string()], 1.0, true)
        })
        .collect();

    let space = GridSearchSpace::new(vec![0.05, 0.1])
        .with_epochs(vec![5])
        .without_early_stopping();

    let result = StrategyPredictor::grid_search_with_cv(&examples, &space, 3);

    assert_eq!(result.total_evaluated, 2);
    assert_eq!(result.k_folds, 3);
    assert_eq!(result.search_method, "grid");
    assert!(result.best_idx < result.results.len());

    // Check best hyperparameters
    let best = result.best_hyperparameters();
    assert!(best.learning_rate == 0.05 || best.learning_rate == 0.1);
}

#[test]
fn test_random_search_with_cv() {
    let properties: Vec<_> = (0..20)
        .map(|i| make_temporal(&format!("rand_cv_temporal_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 2 == 0 {
                BackendId::TlaPlus
            } else {
                BackendId::Alloy
            };
            TrainingExample::from_verification(p, backend, vec!["auto".to_string()], 1.0, true)
        })
        .collect();

    let config = RandomSearchConfig::new(3)
        .with_epoch_range(5, 10)
        .with_seed(42);

    let result = StrategyPredictor::random_search_with_cv(&examples, &config, 2);

    assert_eq!(result.total_evaluated, 3);
    assert_eq!(result.k_folds, 2);
    assert_eq!(result.search_method, "random");
}

#[test]
fn test_cv_search_result_methods() {
    let properties: Vec<_> = (0..15)
        .map(|i| make_invariant(&format!("cv_method_inv_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 3 == 0 {
                BackendId::Kani
            } else if i % 3 == 1 {
                BackendId::Lean4
            } else {
                BackendId::TlaPlus
            };
            TrainingExample::from_verification(p, backend, vec!["auto".to_string()], 1.0, true)
        })
        .collect();

    let space = GridSearchSpace::new(vec![0.01, 0.05, 0.1])
        .with_epochs(vec![5])
        .without_early_stopping();

    let result = StrategyPredictor::grid_search_with_cv(&examples, &space, 2);

    // Test sorted_by_loss
    let sorted = result.sorted_by_loss();
    for i in 1..sorted.len() {
        assert!(sorted[i - 1].mean_val_loss <= sorted[i].mean_val_loss);
    }

    // Test top_n
    let top2 = result.top_n(2);
    assert_eq!(top2.len(), 2);

    // Test best_result
    let best = result.best_result();
    assert_eq!(best.hyperparameters.epochs, 5);
}

// ========================================================================
// Bayesian Optimization Tests
// ========================================================================

#[test]
fn test_bayesian_optimizer_defaults() {
    let optimizer = BayesianOptimizer::default();
    assert_eq!(optimizer.n_iterations, 25);
    assert_eq!(optimizer.n_initial_samples, 5);
    assert!(optimizer.kappa > 0.0);
}

#[test]
fn test_bayesian_optimizer_builder() {
    let optimizer = BayesianOptimizer::new(10)
        .with_lr_bounds(0.001, 0.1)
        .with_epoch_bounds(5, 50)
        .with_initial_samples(3)
        .with_kappa(1.96)
        .with_seed(123);

    assert_eq!(optimizer.n_iterations, 10);
    assert_eq!(optimizer.n_initial_samples, 3);
    assert_eq!(optimizer.lr_bounds, (0.001, 0.1));
    assert_eq!(optimizer.epoch_bounds, (5, 50));
    assert_eq!(optimizer.kappa, 1.96);
    assert_eq!(optimizer.seed, 123);
}

#[test]
fn test_bayesian_hp_to_features() {
    let optimizer = BayesianOptimizer::new(10)
        .with_lr_bounds(0.001, 0.1)
        .with_epoch_bounds(10, 100)
        .with_val_split_bounds(0.1, 0.3);

    let hp = Hyperparameters::new(0.01, 55).with_validation_split(0.2);
    let features = optimizer.hp_to_features(&hp);

    assert_eq!(features.len(), 3);
    for f in &features {
        assert!(*f >= 0.0 && *f <= 1.0, "Feature {} out of bounds", f);
    }
}

#[test]
fn test_bayesian_features_to_hp() {
    let optimizer = BayesianOptimizer::new(10)
        .with_lr_bounds(0.001, 0.1)
        .with_epoch_bounds(10, 100);

    let features = vec![0.5, 0.5, 0.5];
    let hp = optimizer.features_to_hp(&features, LearningRateScheduler::Constant, None);

    assert!(hp.learning_rate > 0.001 && hp.learning_rate < 0.1);
    assert!(hp.epochs >= 10 && hp.epochs <= 100);
}

#[test]
fn test_rbf_kernel() {
    let x1 = vec![0.0, 0.0];
    let x2 = vec![0.0, 0.0];
    let k_same = BayesianOptimizer::rbf_kernel(&x1, &x2, 1.0);
    assert!((k_same - 1.0).abs() < 1e-10);

    let x3 = vec![1.0, 1.0];
    let k_diff = BayesianOptimizer::rbf_kernel(&x1, &x3, 1.0);
    assert!(k_diff < 1.0);
    assert!(k_diff > 0.0);
}

#[test]
fn test_cholesky_decomposition() {
    // Simple 2x2 positive definite matrix
    let matrix = vec![vec![4.0, 2.0], vec![2.0, 5.0]];

    let l = BayesianOptimizer::cholesky(&matrix);
    assert!(l.is_some());

    let l = l.unwrap();
    // Verify L * L^T = A
    let reconstructed_00 = l[0][0] * l[0][0];
    let reconstructed_01 = l[1][0] * l[0][0];
    let reconstructed_11 = l[1][0] * l[1][0] + l[1][1] * l[1][1];

    assert!((reconstructed_00 - 4.0).abs() < 1e-10);
    assert!((reconstructed_01 - 2.0).abs() < 1e-10);
    assert!((reconstructed_11 - 5.0).abs() < 1e-10);
}

#[test]
fn test_cholesky_not_positive_definite() {
    // Not positive definite
    let matrix = vec![vec![1.0, 2.0], vec![2.0, 1.0]];

    let l = BayesianOptimizer::cholesky(&matrix);
    assert!(l.is_none());
}

#[test]
fn test_solve_lower_triangular() {
    let l = vec![vec![2.0, 0.0], vec![1.0, 3.0]];
    let b = vec![4.0, 7.0];

    let x = BayesianOptimizer::solve_lower(&l, &b);

    // Verify L * x = b
    assert!((l[0][0] * x[0] - b[0]).abs() < 1e-10);
    assert!((l[1][0] * x[0] + l[1][1] * x[1] - b[1]).abs() < 1e-10);
}

#[test]
fn test_bayesian_optimize() {
    let properties: Vec<_> = (0..25)
        .map(|i| make_theorem(&format!("bayes_thm_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 3 == 0 {
                BackendId::Lean4
            } else if i % 3 == 1 {
                BackendId::Coq
            } else {
                BackendId::TlaPlus
            };
            TrainingExample::from_verification(p, backend, vec!["auto".to_string()], 1.0, true)
        })
        .collect();

    let optimizer = BayesianOptimizer::new(8)
        .with_initial_samples(3)
        .with_epoch_bounds(5, 15)
        .with_seed(42);

    let result = StrategyPredictor::bayesian_optimize(&examples, &optimizer);

    assert_eq!(result.total_iterations, 8);
    assert_eq!(result.n_initial_samples, 3);
    assert_eq!(result.evaluations.len(), 8);
    assert!(result.best_idx < result.evaluations.len());

    let best = result.best_result();
    assert!(best.val_loss.is_finite());
}

#[test]
fn test_bayesian_optimize_with_cv() {
    let properties: Vec<_> = (0..20)
        .map(|i| make_invariant(&format!("bayes_cv_inv_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let backend = if i % 2 == 0 {
                BackendId::Lean4
            } else {
                BackendId::Kani
            };
            TrainingExample::from_verification(p, backend, vec!["simp".to_string()], 1.0, true)
        })
        .collect();

    let optimizer = BayesianOptimizer::new(5)
        .with_initial_samples(2)
        .with_epoch_bounds(3, 8)
        .with_seed(123);

    let result = StrategyPredictor::bayesian_optimize_with_cv(&examples, &optimizer, 2);

    assert_eq!(result.total_evaluated, 5);
    assert_eq!(result.k_folds, 2);
    assert_eq!(result.search_method, "bayesian");
    assert!(result.best_idx < result.results.len());

    let best = result.best_result();
    assert!(best.mean_val_loss.is_finite());
    assert!(best.std_val_loss >= 0.0);
}

#[test]
fn test_bayesian_result_methods() {
    let properties: Vec<_> = (0..15)
        .map(|i| make_temporal(&format!("bayes_method_temp_{}", i)))
        .collect();
    let examples: Vec<TrainingExample> = properties
        .iter()
        .map(|p| {
            TrainingExample::from_verification(
                p,
                BackendId::TlaPlus,
                vec!["auto".to_string()],
                1.0,
                true,
            )
        })
        .collect();

    let optimizer = BayesianOptimizer::new(5)
        .with_initial_samples(2)
        .with_epoch_bounds(3, 6);

    let result = StrategyPredictor::bayesian_optimize(&examples, &optimizer);

    // Test sorted_by_loss
    let sorted = result.sorted_by_loss();
    for i in 1..sorted.len() {
        assert!(sorted[i - 1].val_loss <= sorted[i].val_loss);
    }

    // Test best_hyperparameters
    let best_hp = result.best_hyperparameters();
    assert!(best_hp.learning_rate > 0.0);
    assert!(best_hp.epochs > 0);
}

#[test]
fn test_ucb_acquisition() {
    // Lower mean + lower variance should give lower UCB
    let ucb1 = BayesianOptimizer::ucb(0.5, 0.1, 2.0);
    let ucb2 = BayesianOptimizer::ucb(0.5, 0.4, 2.0);

    // Higher variance leads to more exploration (lower UCB when minimizing)
    assert!(ucb2 < ucb1);

    // Lower mean gives lower UCB
    let ucb3 = BayesianOptimizer::ucb(0.3, 0.1, 2.0);
    assert!(ucb3 < ucb1);
}
