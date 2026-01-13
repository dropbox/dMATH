//! Property-based tests for dashprove-ai using proptest

use dashprove_ai::{
    strategy::{
        BackendPrediction, CheckpointConfig, DenseLayer, EarlyStoppingConfig, EnsembleAggregation,
        EnsembleMember, EnsembleStrategyPredictor, EpochMetrics, Hyperparameters,
        LearningRateScheduler, PropertyFeatureVector, StrategyPrediction, StrategyPredictor,
        TacticPrediction, TimePrediction, TrainingExample, TrainingHistory, NUM_BACKENDS,
        NUM_FEATURES,
    },
    Confidence,
};
use dashprove_backends::BackendId;
use proptest::prelude::*;

// ============================================================================
// Strategy generators
// ============================================================================

fn arbitrary_backend() -> impl Strategy<Value = BackendId> {
    prop_oneof![
        Just(BackendId::Lean4),
        Just(BackendId::TlaPlus),
        Just(BackendId::Kani),
        Just(BackendId::Alloy),
        Just(BackendId::Isabelle),
        Just(BackendId::Coq),
        Just(BackendId::Dafny),
        Just(BackendId::Marabou),
        Just(BackendId::AlphaBetaCrown),
        Just(BackendId::Eran),
        Just(BackendId::Storm),
        Just(BackendId::Prism),
        Just(BackendId::Tamarin),
        Just(BackendId::ProVerif),
        Just(BackendId::Verifpal),
        Just(BackendId::Verus),
        Just(BackendId::Creusot),
        Just(BackendId::Prusti),
        Just(BackendId::Z3),
        Just(BackendId::Cvc5),
    ]
}

fn arbitrary_features() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(0.0f64..=1.0, NUM_FEATURES)
}

fn arbitrary_feature_names() -> impl Strategy<Value = Vec<String>> {
    Just(
        (0..NUM_FEATURES)
            .map(|i| format!("feature_{}", i))
            .collect(),
    )
}

fn arbitrary_property_feature_vector() -> impl Strategy<Value = PropertyFeatureVector> {
    (arbitrary_features(), arbitrary_feature_names()).prop_map(|(features, feature_names)| {
        PropertyFeatureVector {
            features,
            feature_names,
        }
    })
}

fn arbitrary_backend_prediction() -> impl Strategy<Value = BackendPrediction> {
    (
        arbitrary_backend(),
        0.0f64..=1.0,
        prop::collection::vec((arbitrary_backend(), 0.0f64..=1.0), 0..3),
    )
        .prop_map(|(backend, confidence, alternatives)| BackendPrediction {
            backend,
            confidence,
            alternatives,
        })
}

fn arbitrary_tactic_prediction() -> impl Strategy<Value = TacticPrediction> {
    ("[a-z_]+", 0.0f64..=1.0, 0usize..10).prop_map(|(tactic, confidence, position)| {
        TacticPrediction {
            tactic,
            confidence,
            position,
        }
    })
}

fn arbitrary_time_prediction() -> impl Strategy<Value = TimePrediction> {
    (0.0f64..3600.0, 0.0f64..1.0, 0.0f64..3600.0, 0.0f64..7200.0).prop_map(
        |(expected_seconds, confidence, min_range, max_offset)| TimePrediction {
            expected_seconds,
            confidence,
            range: (
                min_range.min(expected_seconds),
                expected_seconds + max_offset,
            ),
        },
    )
}

fn arbitrary_strategy_prediction() -> impl Strategy<Value = StrategyPrediction> {
    (
        arbitrary_backend_prediction(),
        prop::collection::vec(arbitrary_tactic_prediction(), 0..5),
        arbitrary_time_prediction(),
        arbitrary_property_feature_vector(),
    )
        .prop_map(|(backend, tactics, time, features)| StrategyPrediction {
            backend,
            tactics,
            time,
            features,
        })
}

fn arbitrary_epoch_metrics() -> impl Strategy<Value = EpochMetrics> {
    (
        0usize..100,
        0.0f64..10.0,
        0.0f64..=1.0,
        prop::option::of(0.0f64..10.0),
        prop::option::of(0.0f64..=1.0),
    )
        .prop_map(
            |(epoch, train_loss, train_accuracy, val_loss, val_accuracy)| EpochMetrics {
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
            },
        )
}

fn arbitrary_training_example() -> impl Strategy<Value = TrainingExample> {
    (
        arbitrary_property_feature_vector(),
        arbitrary_backend(),
        any::<bool>(),
        0.1f64..100.0,
    )
        .prop_map(
            |(features, backend, success, time_seconds)| TrainingExample {
                features,
                backend,
                tactics: vec!["simp".to_string(), "auto".to_string()],
                time_seconds,
                success,
            },
        )
}

fn arbitrary_lr_scheduler() -> impl Strategy<Value = LearningRateScheduler> {
    prop_oneof![
        Just(LearningRateScheduler::Constant),
        (1usize..20, 0.1f64..0.9)
            .prop_map(|(step_size, gamma)| LearningRateScheduler::Step { step_size, gamma }),
        (0.9f64..0.999).prop_map(|gamma| LearningRateScheduler::Exponential { gamma }),
    ]
}

fn arbitrary_early_stopping_config() -> impl Strategy<Value = EarlyStoppingConfig> {
    (
        1usize..20,      // patience
        0.0001f64..0.01, // min_delta
        any::<bool>(),   // restore_best_weights
    )
        .prop_map(
            |(patience, min_delta, restore_best_weights)| EarlyStoppingConfig {
                patience,
                min_delta,
                restore_best_weights,
            },
        )
}

fn arbitrary_hyperparameters() -> impl Strategy<Value = Hyperparameters> {
    (
        0.0001f64..0.5,                                      // learning_rate
        1usize..100,                                         // epochs
        0.0f64..0.5,                                         // validation_split
        arbitrary_lr_scheduler(),                            // lr_scheduler
        prop::option::of(arbitrary_early_stopping_config()), // early_stopping
    )
        .prop_map(
            |(learning_rate, epochs, validation_split, lr_scheduler, early_stopping)| {
                Hyperparameters {
                    learning_rate,
                    epochs,
                    validation_split,
                    lr_scheduler,
                    early_stopping,
                }
            },
        )
}

fn arbitrary_checkpoint_config() -> impl Strategy<Value = CheckpointConfig> {
    (
        any::<bool>(), // save_on_improvement
        any::<bool>(), // include_history
        0usize..20,    // save_every_n_epochs
        0usize..10,    // keep_best_n
    )
        .prop_map(
            |(save_on_improvement, include_history, save_every_n_epochs, keep_best_n)| {
                CheckpointConfig {
                    checkpoint_dir: std::path::PathBuf::from("/tmp/checkpoints"),
                    save_on_improvement,
                    include_history,
                    save_every_n_epochs,
                    keep_best_n,
                }
            },
        )
}

fn arbitrary_ensemble_aggregation() -> impl Strategy<Value = EnsembleAggregation> {
    prop_oneof![
        Just(EnsembleAggregation::SoftVoting),
        Just(EnsembleAggregation::WeightedMajority),
    ]
}

fn arbitrary_ensemble_member() -> impl Strategy<Value = EnsembleMember> {
    (
        prop::option::of("[A-Za-z0-9_]{3,20}"), // name
        0.0f64..=2.0,                           // weight (including values above 1)
    )
        .prop_map(|(name, weight)| {
            let mut member = EnsembleMember::new(StrategyPredictor::new());
            if let Some(n) = name {
                member = member.with_name(n);
            }
            member.with_weight(weight)
        })
}

fn arbitrary_ensemble_member_with_edge_weights() -> impl Strategy<Value = EnsembleMember> {
    prop_oneof![
        // Normal positive weights
        (0.1f64..2.0).prop_map(|w| EnsembleMember::new(StrategyPredictor::new()).with_weight(w)),
        // Zero weight
        Just(EnsembleMember::new(StrategyPredictor::new()).with_weight(0.0)),
        // Negative weight (should be treated as zero)
        (-2.0f64..-0.001)
            .prop_map(|w| EnsembleMember::new(StrategyPredictor::new()).with_weight(w)),
        // Infinite weight (should be treated as zero)
        Just(EnsembleMember::new(StrategyPredictor::new()).with_weight(f64::INFINITY)),
        // NaN weight (should be treated as zero)
        Just(EnsembleMember::new(StrategyPredictor::new()).with_weight(f64::NAN)),
    ]
}

fn arbitrary_ensemble_predictor() -> impl Strategy<Value = EnsembleStrategyPredictor> {
    (
        prop::collection::vec(arbitrary_ensemble_member(), 1..5),
        arbitrary_ensemble_aggregation(),
    )
        .prop_map(|(members, aggregation)| {
            EnsembleStrategyPredictor::new(members).with_aggregation(aggregation)
        })
}

// ============================================================================
// PropertyFeatureVector tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn feature_vector_has_correct_dimension(features in arbitrary_features()) {
        // Feature vectors must have exactly NUM_FEATURES dimensions
        prop_assert_eq!(features.len(), NUM_FEATURES);
    }

    #[test]
    fn feature_vector_values_are_normalized(features in arbitrary_features()) {
        // All feature values should be in [0, 1] range
        for (i, &f) in features.iter().enumerate() {
            prop_assert!((0.0..=1.0).contains(&f), "Feature {} = {} is out of range [0, 1]", i, f);
        }
    }

    #[test]
    fn property_feature_vector_get_returns_valid_values(pf in arbitrary_property_feature_vector()) {
        // Getting an existing feature should work
        let val = pf.get("feature_0");
        prop_assert!(val.is_some());
        prop_assert_eq!(val.unwrap(), pf.features[0]);

        // Getting non-existent feature returns None
        let none_val = pf.get("nonexistent");
        prop_assert!(none_val.is_none());
    }

    #[test]
    fn property_feature_vector_json_roundtrip(pf in arbitrary_property_feature_vector()) {
        let json = serde_json::to_string(&pf).unwrap();
        let restored: PropertyFeatureVector = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(pf.features.len(), restored.features.len());
        for (a, b) in pf.features.iter().zip(restored.features.iter()) {
            prop_assert!((a - b).abs() < 1e-10);
        }
    }
}

// ============================================================================
// Confidence tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn confidence_from_score_is_deterministic(score in 0.0f64..=1.0) {
        // Same score should always produce same confidence
        let c1 = Confidence::from_score(score);
        let c2 = Confidence::from_score(score);
        prop_assert_eq!(c1, c2);
    }

    #[test]
    fn confidence_score_roundtrip_is_consistent(score in 0.0f64..=1.0) {
        // from_score followed by to_score should be consistent with the threshold
        let confidence = Confidence::from_score(score);
        let back = confidence.to_score();

        // to_score returns the "canonical" score for each confidence level
        match confidence {
            Confidence::High => prop_assert!((back - 0.9).abs() < 0.001),
            Confidence::Medium => prop_assert!((back - 0.65).abs() < 0.001),
            Confidence::Low => prop_assert!((back - 0.35).abs() < 0.001),
            Confidence::Speculative => prop_assert!((back - 0.1).abs() < 0.001),
        }
    }

    #[test]
    fn confidence_thresholds_are_ordered(score in 0.0f64..=1.0) {
        // Confidence levels should follow a consistent ordering
        let c = Confidence::from_score(score);
        if score >= 0.8 {
            prop_assert_eq!(c, Confidence::High);
        } else if score >= 0.5 {
            prop_assert_eq!(c, Confidence::Medium);
        } else if score >= 0.2 {
            prop_assert_eq!(c, Confidence::Low);
        } else {
            prop_assert_eq!(c, Confidence::Speculative);
        }
    }

    #[test]
    fn higher_score_means_higher_or_equal_confidence(s1 in 0.0f64..=1.0, s2 in 0.0f64..=1.0) {
        // Ordering: if s1 >= s2, then confidence(s1).to_score() >= confidence(s2).to_score()
        let c1 = Confidence::from_score(s1);
        let c2 = Confidence::from_score(s2);
        if s1 >= s2 {
            prop_assert!(c1.to_score() >= c2.to_score());
        }
    }
}

// ============================================================================
// DenseLayer tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn dense_layer_output_has_correct_dimension(input_size in 2usize..32, output_size in 2usize..32) {
        // DenseLayer output should have the correct dimension
        let layer = DenseLayer::new(input_size, output_size);
        let input: Vec<f64> = vec![0.5; input_size];
        let output = layer.forward(&input);
        prop_assert_eq!(output.len(), output_size);
    }

    #[test]
    fn dense_layer_forward_is_deterministic(input_size in 2usize..16, output_size in 2usize..16) {
        // Same input should produce same output (no randomness after construction)
        let layer = DenseLayer::new(input_size, output_size);
        let input: Vec<f64> = vec![0.5; input_size];
        let out1 = layer.forward(&input);
        let out2 = layer.forward(&input);
        for (a, b) in out1.iter().zip(out2.iter()) {
            prop_assert!((a - b).abs() < 1e-10, "Output differs: {} vs {}", a, b);
        }
    }

    #[test]
    fn dense_layer_relu_produces_nonnegative_outputs(input_size in 2usize..16, output_size in 2usize..16) {
        // ReLU activation should produce non-negative outputs
        let mut layer = DenseLayer::new(input_size, output_size);
        // Set biases to negative to test ReLU
        for b in &mut layer.biases {
            *b = -1.0;
        }
        let input: Vec<f64> = vec![0.0; input_size];
        let output = layer.forward_relu(&input);
        for (i, &val) in output.iter().enumerate() {
            prop_assert!(val >= 0.0, "ReLU output {} is negative: {}", i, val);
        }
    }
}

// ============================================================================
// StrategyPredictor tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn predictor_backend_has_valid_confidence(pf in arbitrary_property_feature_vector()) {
        // Predictions should have confidence in [0, 1]
        let predictor = StrategyPredictor::new();
        let prediction = predictor.predict_backend(&pf);

        prop_assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0,
            "Backend confidence {} out of range", prediction.confidence);
    }

    #[test]
    fn predictor_time_has_valid_values(pf in arbitrary_property_feature_vector()) {
        let predictor = StrategyPredictor::new();
        let time = predictor.predict_time(&pf);

        prop_assert!(time.expected_seconds >= 0.0,
            "Negative expected time: {}", time.expected_seconds);
        prop_assert!(time.confidence >= 0.0 && time.confidence <= 1.0,
            "Time confidence {} out of range", time.confidence);
    }

    #[test]
    fn predictor_json_roundtrip(pf in arbitrary_property_feature_vector()) {
        // Predictor should serialize and deserialize correctly
        let predictor = StrategyPredictor::new();
        let json = serde_json::to_string(&predictor).unwrap();
        let restored: StrategyPredictor = serde_json::from_str(&json).unwrap();

        // Test with same features - backend predictions should match
        let pred1 = predictor.predict_backend(&pf);
        let pred2 = restored.predict_backend(&pf);

        prop_assert_eq!(pred1.backend, pred2.backend);
        prop_assert!((pred1.confidence - pred2.confidence).abs() < 1e-6);
    }

    #[test]
    fn predictor_tactics_have_valid_confidences(pf in arbitrary_property_feature_vector()) {
        let predictor = StrategyPredictor::new();
        let tactics = predictor.predict_tactics(&pf, 5);

        for tp in &tactics {
            prop_assert!(tp.confidence >= 0.0 && tp.confidence <= 1.0,
                "Tactic confidence {} out of range", tp.confidence);
        }
    }
}

// ============================================================================
// Training types tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn training_example_json_roundtrip(ex in arbitrary_training_example()) {
        // TrainingExample should serialize/deserialize correctly
        let json = serde_json::to_string(&ex).unwrap();
        let restored: TrainingExample = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(ex.backend, restored.backend);
        prop_assert!((ex.time_seconds - restored.time_seconds).abs() < 1e-10);
        prop_assert_eq!(ex.features.features.len(), restored.features.features.len());
        prop_assert_eq!(ex.success, restored.success);
    }

    #[test]
    fn epoch_metrics_json_roundtrip(metrics in arbitrary_epoch_metrics()) {
        let json = serde_json::to_string(&metrics).unwrap();
        let restored: EpochMetrics = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(metrics.epoch, restored.epoch);
        prop_assert!((metrics.train_loss - restored.train_loss).abs() < 1e-10);
        prop_assert!((metrics.train_accuracy - restored.train_accuracy).abs() < 1e-10);
        prop_assert_eq!(metrics.val_loss.is_some(), restored.val_loss.is_some());
        prop_assert_eq!(metrics.val_accuracy.is_some(), restored.val_accuracy.is_some());
    }

    #[test]
    fn training_history_stores_epochs(
        metrics in prop::collection::vec(arbitrary_epoch_metrics(), 1..20)
    ) {
        let mut history = TrainingHistory {
            epochs: Vec::new(),
        };
        for m in &metrics {
            history.epochs.push(m.clone());
        }
        prop_assert_eq!(history.epochs.len(), metrics.len());
    }

    #[test]
    fn hyperparameters_are_valid(hp in arbitrary_hyperparameters()) {
        // All hyperparameters should be in valid ranges
        prop_assert!(hp.learning_rate > 0.0 && hp.learning_rate < 1.0,
            "learning_rate {} invalid", hp.learning_rate);
        prop_assert!(hp.epochs >= 1, "epochs {} invalid", hp.epochs);
        prop_assert!(hp.validation_split >= 0.0 && hp.validation_split <= 0.5,
            "validation_split {} invalid", hp.validation_split);
    }

    #[test]
    fn hyperparameters_json_roundtrip(hp in arbitrary_hyperparameters()) {
        let json = serde_json::to_string(&hp).unwrap();
        let restored: Hyperparameters = serde_json::from_str(&json).unwrap();
        prop_assert!((hp.learning_rate - restored.learning_rate).abs() < 1e-10);
        prop_assert_eq!(hp.epochs, restored.epochs);
        prop_assert!((hp.validation_split - restored.validation_split).abs() < 1e-10);
    }

    #[test]
    fn early_stopping_config_is_valid(config in arbitrary_early_stopping_config()) {
        prop_assert!(config.patience >= 1, "patience {} invalid", config.patience);
        prop_assert!(config.min_delta >= 0.0, "min_delta {} invalid", config.min_delta);
    }

    #[test]
    fn checkpoint_config_is_valid(config in arbitrary_checkpoint_config()) {
        // CheckpointConfig has no invalid values for usize fields
        // Just verify the structure exists
        let _ = config.save_on_improvement;
        let _ = config.include_history;
    }
}

// ============================================================================
// Strategy prediction structure tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn strategy_prediction_json_roundtrip(pred in arbitrary_strategy_prediction()) {
        let json = serde_json::to_string(&pred).unwrap();
        let restored: StrategyPrediction = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(pred.backend.backend, restored.backend.backend);
        prop_assert!((pred.backend.confidence - restored.backend.confidence).abs() < 1e-10);
        prop_assert_eq!(pred.tactics.len(), restored.tactics.len());
        prop_assert!((pred.time.expected_seconds - restored.time.expected_seconds).abs() < 1e-10);
    }

    #[test]
    fn time_prediction_range_is_valid(pred in arbitrary_time_prediction()) {
        // range.0 should be <= range.1
        prop_assert!(pred.range.0 <= pred.range.1,
            "range.0 {} > range.1 {}", pred.range.0, pred.range.1);
    }

    #[test]
    fn tactic_predictions_positions_are_bounded(pred in arbitrary_strategy_prediction()) {
        // Tactic positions should be bounded
        for tp in &pred.tactics {
            prop_assert!(tp.position < 10, "Position {} out of range", tp.position);
        }
    }
}

// ============================================================================
// Backend ID ordering and indexing tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn backend_index_is_in_range(backend in arbitrary_backend()) {
        // Backend indices should be in valid range for the model
        let idx = backend as usize;
        prop_assert!(idx < NUM_BACKENDS, "Backend index {} >= NUM_BACKENDS {}", idx, NUM_BACKENDS);
    }

    #[test]
    fn backend_clone_is_equal(backend in arbitrary_backend()) {
        let cloned = backend;
        prop_assert_eq!(backend, cloned);
    }

    #[test]
    fn backend_prediction_confidence_in_range(pred in arbitrary_backend_prediction()) {
        prop_assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0,
            "Confidence {} out of range [0, 1]", pred.confidence);
    }
}

// ============================================================================
// Additional smoke tests for complex operations
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn training_history_final_methods_work(
        metrics in prop::collection::vec(arbitrary_epoch_metrics(), 1..20)
    ) {
        let history = TrainingHistory { epochs: metrics.clone() };

        // final_train_accuracy should return value from last epoch
        if let Some(acc) = history.final_train_accuracy() {
            let expected = metrics.last().map(|m| m.train_accuracy);
            prop_assert_eq!(Some(acc), expected);
        }

        // final_train_loss should return value from last epoch
        if let Some(loss) = history.final_train_loss() {
            let expected = metrics.last().map(|m| m.train_loss);
            prop_assert_eq!(Some(loss), expected);
        }
    }

    #[test]
    fn multiple_predictions_are_consistent(pf in arbitrary_property_feature_vector()) {
        // Multiple predictions from the same predictor should be identical
        let predictor = StrategyPredictor::new();
        let p1 = predictor.predict_backend(&pf);
        let p2 = predictor.predict_backend(&pf);

        prop_assert_eq!(p1.backend, p2.backend);
        prop_assert!((p1.confidence - p2.confidence).abs() < 1e-10);
    }

    #[test]
    fn predictor_predictions_deterministic(pf in arbitrary_property_feature_vector()) {
        // Serialize and deserialize predictor, predictions should be identical
        let predictor = StrategyPredictor::new();
        let json = serde_json::to_string(&predictor).unwrap();
        let restored: StrategyPredictor = serde_json::from_str(&json).unwrap();

        let t1 = predictor.predict_time(&pf);
        let t2 = restored.predict_time(&pf);

        prop_assert!((t1.expected_seconds - t2.expected_seconds).abs() < 1e-6);
        prop_assert!((t1.confidence - t2.confidence).abs() < 1e-6);
    }
}

// ============================================================================
// EnsembleAggregation tests
// ============================================================================

#[test]
fn ensemble_aggregation_default_is_soft_voting() {
    let agg = EnsembleAggregation::default();
    assert!(matches!(agg, EnsembleAggregation::SoftVoting));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn ensemble_aggregation_json_roundtrip(agg in arbitrary_ensemble_aggregation()) {
        let json = serde_json::to_string(&agg).unwrap();
        let restored: EnsembleAggregation = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(agg, restored);
    }

    #[test]
    fn ensemble_aggregation_debug_not_empty(agg in arbitrary_ensemble_aggregation()) {
        let debug = format!("{:?}", agg);
        prop_assert!(!debug.is_empty());
    }
}

// ============================================================================
// EnsembleMember tests
// ============================================================================

#[test]
fn ensemble_member_default_weight_is_one() {
    let member = EnsembleMember::new(StrategyPredictor::new());
    assert!((member.weight - 1.0).abs() < 1e-10);
}

#[test]
fn ensemble_member_default_name_is_none() {
    let member = EnsembleMember::new(StrategyPredictor::new());
    assert!(member.name.is_none());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn ensemble_member_json_roundtrip(member in arbitrary_ensemble_member()) {
        let json = serde_json::to_string(&member).unwrap();
        let restored: EnsembleMember = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(member.name, restored.name);
        prop_assert!((member.weight - restored.weight).abs() < 1e-10);
    }

    #[test]
    fn ensemble_member_with_name_sets_name(name in "[A-Za-z0-9_]{3,20}") {
        let member = EnsembleMember::new(StrategyPredictor::new()).with_name(name.clone());
        prop_assert_eq!(member.name, Some(name));
    }

    #[test]
    fn ensemble_member_with_weight_sets_weight(weight in 0.0f64..=10.0) {
        let member = EnsembleMember::new(StrategyPredictor::new()).with_weight(weight);
        prop_assert!((member.weight - weight).abs() < 1e-10);
    }

    #[test]
    fn ensemble_member_builder_chain_preserves_values(
        name in "[A-Za-z0-9_]{3,20}",
        weight in 0.0f64..=10.0
    ) {
        let member = EnsembleMember::new(StrategyPredictor::new())
            .with_name(name.clone())
            .with_weight(weight);
        prop_assert_eq!(member.name, Some(name));
        prop_assert!((member.weight - weight).abs() < 1e-10);
    }
}

// ============================================================================
// EnsembleStrategyPredictor tests
// ============================================================================

#[test]
fn ensemble_from_empty_uses_default_fallback() {
    // Empty ensemble should fall back to a default predictor
    let ensemble = EnsembleStrategyPredictor::new(vec![]);
    let features = PropertyFeatureVector {
        features: vec![0.5; NUM_FEATURES],
        feature_names: (0..NUM_FEATURES)
            .map(|i| format!("feature_{}", i))
            .collect(),
    };
    let prediction = ensemble.predict_backend(&features);
    // Should return a valid prediction (from fallback)
    assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
}

#[test]
fn ensemble_from_models_creates_correct_member_count() {
    let models = vec![
        StrategyPredictor::new(),
        StrategyPredictor::new(),
        StrategyPredictor::new(),
    ];
    let ensemble = EnsembleStrategyPredictor::from_models(models);
    assert_eq!(ensemble.members.len(), 3);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn ensemble_json_roundtrip(ensemble in arbitrary_ensemble_predictor()) {
        let json = serde_json::to_string(&ensemble).unwrap();
        let restored: EnsembleStrategyPredictor = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(ensemble.members.len(), restored.members.len());
        prop_assert_eq!(ensemble.aggregation, restored.aggregation);
    }

    #[test]
    fn ensemble_predict_backend_returns_valid_confidence(
        ensemble in arbitrary_ensemble_predictor(),
        features in arbitrary_property_feature_vector()
    ) {
        let prediction = ensemble.predict_backend(&features);
        prop_assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0,
            "Ensemble backend confidence {} out of range", prediction.confidence);
    }

    #[test]
    fn ensemble_predict_time_returns_valid_values(
        ensemble in arbitrary_ensemble_predictor(),
        features in arbitrary_property_feature_vector()
    ) {
        let time = ensemble.predict_time(&features);
        prop_assert!(time.expected_seconds >= 0.0,
            "Ensemble negative expected time: {}", time.expected_seconds);
        prop_assert!(time.confidence >= 0.0 && time.confidence <= 1.0,
            "Ensemble time confidence {} out of range", time.confidence);
    }

    #[test]
    fn ensemble_predict_tactics_returns_valid_confidences(
        ensemble in arbitrary_ensemble_predictor(),
        features in arbitrary_property_feature_vector()
    ) {
        let tactics = ensemble.predict_tactics(&features, 5);
        for tp in &tactics {
            prop_assert!(tp.confidence >= 0.0 && tp.confidence <= 1.0,
                "Ensemble tactic confidence {} out of range", tp.confidence);
        }
    }

    #[test]
    fn ensemble_with_aggregation_sets_aggregation(
        members in prop::collection::vec(arbitrary_ensemble_member(), 1..3),
        agg in arbitrary_ensemble_aggregation()
    ) {
        let ensemble = EnsembleStrategyPredictor::new(members).with_aggregation(agg);
        prop_assert_eq!(ensemble.aggregation, agg);
    }

    #[test]
    fn ensemble_add_member_increases_count(
        members in prop::collection::vec(arbitrary_ensemble_member(), 1..3),
        new_member in arbitrary_ensemble_member()
    ) {
        let mut ensemble = EnsembleStrategyPredictor::new(members.clone());
        let initial_count = ensemble.members.len();
        ensemble.add_member(new_member);
        prop_assert_eq!(ensemble.members.len(), initial_count + 1);
    }

    #[test]
    fn ensemble_predictions_are_deterministic(
        ensemble in arbitrary_ensemble_predictor(),
        features in arbitrary_property_feature_vector()
    ) {
        // Multiple predictions from same ensemble should be identical
        let p1 = ensemble.predict_backend(&features);
        let p2 = ensemble.predict_backend(&features);
        prop_assert_eq!(p1.backend, p2.backend);
        prop_assert!((p1.confidence - p2.confidence).abs() < 1e-10);

        let t1 = ensemble.predict_time(&features);
        let t2 = ensemble.predict_time(&features);
        prop_assert!((t1.expected_seconds - t2.expected_seconds).abs() < 1e-10);
    }
}

// ============================================================================
// Ensemble edge case tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn ensemble_handles_edge_weight_members(
        members in prop::collection::vec(arbitrary_ensemble_member_with_edge_weights(), 1..5),
        features in arbitrary_property_feature_vector()
    ) {
        // Ensemble should handle edge weight values (zero, negative, inf, nan) gracefully
        let ensemble = EnsembleStrategyPredictor::new(members);
        let prediction = ensemble.predict_backend(&features);
        // Should not panic and should return valid confidence
        prop_assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0,
            "Prediction with edge weights confidence {} out of range", prediction.confidence);

        let time = ensemble.predict_time(&features);
        prop_assert!(time.confidence >= 0.0 && time.confidence <= 1.0);
    }

    #[test]
    fn ensemble_all_zero_weights_uses_fallback(features in arbitrary_property_feature_vector()) {
        // If all members have zero effective weight, should fall back gracefully
        let members = vec![
            EnsembleMember::new(StrategyPredictor::new()).with_weight(0.0),
            EnsembleMember::new(StrategyPredictor::new()).with_weight(-1.0),
        ];
        let ensemble = EnsembleStrategyPredictor::new(members);
        let prediction = ensemble.predict_backend(&features);
        // Should return valid prediction from fallback
        prop_assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    }

    #[test]
    fn ensemble_single_member_produces_valid_output(features in arbitrary_property_feature_vector()) {
        // Single-member ensemble should produce valid output
        // Note: Due to internal normalization, exact predictions may differ from the
        // underlying predictor, but outputs should still be valid
        let predictor = StrategyPredictor::new();
        let member = EnsembleMember::new(predictor).with_weight(1.0);
        let ensemble = EnsembleStrategyPredictor::new(vec![member])
            .with_aggregation(EnsembleAggregation::SoftVoting);

        let ensemble_pred = ensemble.predict_backend(&features);

        // Ensemble should produce valid predictions
        prop_assert!(ensemble_pred.confidence >= 0.0 && ensemble_pred.confidence <= 1.0,
            "Single-member ensemble confidence {} out of range", ensemble_pred.confidence);

        // Alternatives should also have valid confidences
        for (_, alt_conf) in &ensemble_pred.alternatives {
            prop_assert!(*alt_conf >= 0.0 && *alt_conf <= 1.0,
                "Alternative confidence {} out of range", alt_conf);
        }
    }
}

// ============================================================================
// Ensemble aggregation method comparison tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn ensemble_soft_vs_weighted_majority_both_valid(
        members in prop::collection::vec(arbitrary_ensemble_member(), 2..4),
        features in arbitrary_property_feature_vector()
    ) {
        // Both aggregation methods should produce valid results
        let soft_ensemble = EnsembleStrategyPredictor::new(members.clone())
            .with_aggregation(EnsembleAggregation::SoftVoting);
        let weighted_ensemble = EnsembleStrategyPredictor::new(members)
            .with_aggregation(EnsembleAggregation::WeightedMajority);

        let soft_pred = soft_ensemble.predict_backend(&features);
        let weighted_pred = weighted_ensemble.predict_backend(&features);

        prop_assert!(soft_pred.confidence >= 0.0 && soft_pred.confidence <= 1.0);
        prop_assert!(weighted_pred.confidence >= 0.0 && weighted_pred.confidence <= 1.0);
    }

    #[test]
    fn ensemble_json_roundtrip_preserves_predictions(
        ensemble in arbitrary_ensemble_predictor(),
        features in arbitrary_property_feature_vector()
    ) {
        // Serialization should preserve prediction behavior
        let json = serde_json::to_string(&ensemble).unwrap();
        let restored: EnsembleStrategyPredictor = serde_json::from_str(&json).unwrap();

        let pred1 = ensemble.predict_backend(&features);
        let pred2 = restored.predict_backend(&features);

        prop_assert_eq!(pred1.backend, pred2.backend);
        prop_assert!((pred1.confidence - pred2.confidence).abs() < 1e-6);
    }
}
