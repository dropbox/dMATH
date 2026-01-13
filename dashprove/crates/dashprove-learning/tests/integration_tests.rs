//! Integration tests for dashprove-learning
//!
//! These tests verify the public API of the learning crate including:
//! - ProofLearningSystem creation and basic operations
//! - Proof corpus management and similarity search
//! - Counterexample storage and classification
//! - Tactic database and suggestion system
//! - Persistence and loading
//! - Embedding and similarity search

use dashprove_backends::traits::{
    BackendId, CounterexampleClusters, CounterexampleValue, FailedCheck, StructuredCounterexample,
    TraceState, VerificationStatus,
};
use dashprove_learning::{
    CounterexampleCorpus, EmbeddingIndexBuilder, LearnableResult, ProofCorpus, ProofLearningSystem,
    PropertyEmbedder, PropertyFeatures, TacticContext, TacticDatabase, TimePeriod, EMBEDDING_DIM,
};
use dashprove_usl::ast::{Expr, Invariant, Property, Theorem};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// =============================================================================
// Test Fixtures
// =============================================================================

fn make_invariant(name: &str) -> Property {
    Property::Invariant(Invariant {
        name: name.to_string(),
        body: Expr::Bool(true),
    })
}

fn make_theorem(name: &str) -> Property {
    Property::Theorem(Theorem {
        name: name.to_string(),
        body: Expr::Bool(true),
    })
}

fn make_learnable_result(name: &str, backend: BackendId, tactics: Vec<&str>) -> LearnableResult {
    LearnableResult {
        property: make_invariant(name),
        backend,
        status: VerificationStatus::Proven,
        tactics: tactics.into_iter().map(String::from).collect(),
        time_taken: Duration::from_millis(100),
        proof_output: Some(format!("proof for {}", name)),
    }
}

fn make_counterexample(vars: &[(&str, i128)], description: &str) -> StructuredCounterexample {
    let mut cx = StructuredCounterexample::new();

    for (name, value) in vars {
        cx.witness.insert(
            name.to_string(),
            CounterexampleValue::Int {
                value: *value,
                type_hint: None,
            },
        );
    }

    if !description.is_empty() {
        cx.failed_checks.push(FailedCheck {
            check_id: "test".to_string(),
            description: description.to_string(),
            location: None,
            function: None,
        });
    }

    // Add trace
    let mut state = TraceState::new(1);
    state.action = Some("Init".to_string());
    cx.trace.push(state);

    cx
}

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let mut dir = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    dir.push(format!("dashprove_learning_test_{prefix}_{ts}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// =============================================================================
// ProofLearningSystem Tests
// =============================================================================

#[test]
fn test_learning_system_new() {
    let system = ProofLearningSystem::new();
    assert_eq!(system.proof_count(), 0);
    assert_eq!(system.counterexample_count(), 0);
}

#[test]
fn test_learning_system_default() {
    let system = ProofLearningSystem::default();
    assert_eq!(system.proof_count(), 0);
}

#[test]
fn test_record_proven_result() {
    let mut system = ProofLearningSystem::new();

    let result = make_learnable_result("test_property", BackendId::Lean4, vec!["simp", "decide"]);
    system.record(&result);

    assert_eq!(system.proof_count(), 1);
}

#[test]
fn test_record_multiple_proofs() {
    let mut system = ProofLearningSystem::new();

    for i in 0..10 {
        let result =
            make_learnable_result(&format!("property_{}", i), BackendId::Lean4, vec!["decide"]);
        system.record(&result);
    }

    assert_eq!(system.proof_count(), 10);
}

#[test]
fn test_record_disproven_result() {
    let mut system = ProofLearningSystem::new();

    let result = LearnableResult {
        property: make_invariant("failing_prop"),
        backend: BackendId::Lean4,
        status: VerificationStatus::Disproven,
        tactics: vec!["simp".to_string()],
        time_taken: Duration::from_millis(50),
        proof_output: None,
    };

    system.record(&result);

    // Disproven results don't add to proof corpus
    assert_eq!(system.proof_count(), 0);
}

#[test]
fn test_record_unknown_result() {
    let mut system = ProofLearningSystem::new();

    let result = LearnableResult {
        property: make_invariant("timeout_prop"),
        backend: BackendId::Lean4,
        status: VerificationStatus::Unknown {
            reason: "timeout".to_string(),
        },
        tactics: vec!["simp".to_string()],
        time_taken: Duration::from_secs(300),
        proof_output: None,
    };

    system.record(&result);

    // Unknown results don't add to proof corpus
    assert_eq!(system.proof_count(), 0);
}

#[test]
fn test_record_partial_result() {
    let mut system = ProofLearningSystem::new();

    let result = LearnableResult {
        property: make_invariant("partial_prop"),
        backend: BackendId::Lean4,
        status: VerificationStatus::Partial {
            verified_percentage: 0.75,
        },
        tactics: vec!["simp".to_string()],
        time_taken: Duration::from_millis(200),
        proof_output: None,
    };

    system.record(&result);

    // Partial results don't add to proof corpus (only tracks tactics)
    assert_eq!(system.proof_count(), 0);
}

#[test]
fn test_find_similar_proofs() {
    let mut system = ProofLearningSystem::new();

    // Record some proofs
    for i in 0..5 {
        let result = make_learnable_result(&format!("inv_{}", i), BackendId::Lean4, vec!["decide"]);
        system.record(&result);
    }

    let query = make_invariant("new_inv");
    let similar = system.find_similar(&query, 3);

    // Should return at most k results
    assert!(similar.len() <= 3);
}

#[test]
fn test_find_similar_with_empty_corpus() {
    let system = ProofLearningSystem::new();

    let query = make_invariant("test");
    let similar = system.find_similar(&query, 5);

    assert!(similar.is_empty());
}

#[test]
fn test_search_by_keywords() {
    let mut system = ProofLearningSystem::new();

    // Record proofs with descriptive names
    system.record(&make_learnable_result(
        "buffer_overflow_safety",
        BackendId::Lean4,
        vec!["decide"],
    ));
    system.record(&make_learnable_result(
        "null_pointer_check",
        BackendId::Lean4,
        vec!["decide"],
    ));
    system.record(&make_learnable_result(
        "integer_overflow",
        BackendId::Lean4,
        vec!["decide"],
    ));

    let results = system.search_by_keywords("overflow", 5);

    assert!(!results.is_empty());
    // Should find proofs with "overflow" in name
}

#[test]
fn test_suggest_tactics() {
    let mut system = ProofLearningSystem::new();

    // Record proofs with various tactics
    for _ in 0..5 {
        system.record(&make_learnable_result(
            "inv",
            BackendId::Lean4,
            vec!["simp", "decide"],
        ));
    }

    let query = make_invariant("query");
    let suggestions = system.suggest_tactics(&query, 3);

    assert!(!suggestions.is_empty());
    // Each suggestion is (tactic_name, score)
    for (name, score) in &suggestions {
        assert!(!name.is_empty());
        assert!(*score >= 0.0);
    }
}

#[test]
fn test_get_proof_by_id() {
    let mut system = ProofLearningSystem::new();

    let result = make_learnable_result("test_prop", BackendId::Lean4, vec!["decide"]);
    system.record(&result);

    // Get the first proof (we know there's exactly one)
    let similar = system.find_similar(&make_invariant("test_prop"), 1);
    assert!(!similar.is_empty());

    // SimilarProof has `id` field, not `proof_id`
    let proof = system.get_proof(&similar[0].id);
    assert!(proof.is_some());
    assert_eq!(proof.unwrap().property.name(), "test_prop");
}

#[test]
fn test_proof_history() {
    let mut system = ProofLearningSystem::new();

    for i in 0..5 {
        system.record(&make_learnable_result(
            &format!("prop_{}", i),
            BackendId::Lean4,
            vec!["decide"],
        ));
    }

    // TimePeriod only has Day, Week, Month (no Hour)
    let history = system.proof_history(TimePeriod::Day);
    assert_eq!(history.total_count, 5);
}

// =============================================================================
// Counterexample Tests
// =============================================================================

#[test]
fn test_record_counterexample() {
    let mut system = ProofLearningSystem::new();

    let cx = make_counterexample(&[("x", 5)], "division by zero");
    let id = system.record_counterexample("test_prop", BackendId::TlaPlus, cx, None);

    assert_eq!(system.counterexample_count(), 1);
    assert!(system.get_counterexample(&id).is_some());
}

#[test]
fn test_record_counterexample_with_label() {
    let mut system = ProofLearningSystem::new();

    let cx = make_counterexample(&[("y", 10)], "overflow");
    let id = system.record_counterexample(
        "overflow_prop",
        BackendId::Kani,
        cx,
        Some("integer_overflow_cluster".to_string()),
    );

    let entry = system.get_counterexample(&id).unwrap();
    assert_eq!(
        entry.cluster_label,
        Some("integer_overflow_cluster".to_string())
    );
}

#[test]
fn test_find_similar_counterexamples() {
    let mut system = ProofLearningSystem::new();

    // Record several counterexamples
    for i in 0..5 {
        let cx = make_counterexample(&[("n", i)], "error");
        system.record_counterexample(&format!("prop_{}", i), BackendId::TlaPlus, cx, None);
    }

    let query = make_counterexample(&[("n", 3)], "error");
    let similar = system.find_similar_counterexamples(&query, 3);

    assert!(similar.len() <= 3);
    for s in &similar {
        assert!(s.similarity >= 0.0 && s.similarity <= 1.0);
    }
}

#[test]
fn test_classify_counterexample() {
    let mut system = ProofLearningSystem::new();

    // Create cluster patterns
    let cx1 = make_counterexample(&[("x", 1)], "overflow");
    let cx2 = make_counterexample(&[("x", 2)], "overflow");
    let clusters = CounterexampleClusters::from_counterexamples(vec![cx1, cx2], 0.5);

    system.record_cluster_patterns(&clusters);
    assert!(system.cluster_pattern_count() > 0);

    // Classify similar counterexample
    let query = make_counterexample(&[("x", 3)], "overflow");
    let classification = system.classify_counterexample(&query);

    // Should classify into one of the patterns
    assert!(classification.is_some());
}

#[test]
fn test_search_counterexamples_by_keywords() {
    let mut system = ProofLearningSystem::new();

    let cx1 = make_counterexample(&[("x", 1)], "buffer overflow detected");
    system.record_counterexample("prop1", BackendId::Kani, cx1, None);

    let cx2 = make_counterexample(&[("y", 2)], "null pointer dereference");
    system.record_counterexample("prop2", BackendId::Kani, cx2, None);

    let results = system.search_counterexamples_by_keywords("overflow", 5);
    assert!(!results.is_empty());
}

// =============================================================================
// Tactic Database Tests
// =============================================================================

#[test]
fn test_tactic_database_new() {
    let db = TacticDatabase::new();
    assert_eq!(db.total_observations(), 0);
}

#[test]
fn test_tactic_record_success() {
    let mut db = TacticDatabase::new();

    let context = TacticContext::from_features(&PropertyFeatures::default());
    db.record_success(&context, "simp");

    assert!(db.total_observations() > 0);
}

#[test]
fn test_tactic_record_failure() {
    let mut db = TacticDatabase::new();

    let context = TacticContext::from_features(&PropertyFeatures::default());
    db.record_failure(&context, "simp");

    assert!(db.total_observations() > 0);
}

#[test]
fn test_tactic_record_partial() {
    let mut db = TacticDatabase::new();

    let context = TacticContext::from_features(&PropertyFeatures::default());
    db.record_partial(&context, "simp");

    assert!(db.total_observations() > 0);
}

#[test]
fn test_tactic_best_for_context() {
    let mut db = TacticDatabase::new();

    let context = TacticContext::from_features(&PropertyFeatures::default());

    // Record several successes for "decide"
    for _ in 0..10 {
        db.record_success(&context, "decide");
    }

    // Record some for "simp"
    for _ in 0..3 {
        db.record_success(&context, "simp");
    }

    let best = db.best_for_context(&context, 5);
    assert!(!best.is_empty());

    // "decide" should rank higher than "simp" due to more successes
    let decide_idx = best.iter().position(|(name, _)| name == "decide");
    let simp_idx = best.iter().position(|(name, _)| name == "simp");

    if let (Some(d), Some(s)) = (decide_idx, simp_idx) {
        assert!(d < s, "decide should rank higher than simp");
    }
}

// =============================================================================
// ProofCorpus Tests
// =============================================================================

#[test]
fn test_proof_corpus_new() {
    let corpus = ProofCorpus::new();
    assert_eq!(corpus.len(), 0);
    assert!(corpus.is_empty());
}

#[test]
fn test_proof_corpus_insert() {
    let mut corpus = ProofCorpus::new();

    let result = make_learnable_result("test", BackendId::Lean4, vec!["decide"]);
    let id = corpus.insert(&result);

    assert_eq!(corpus.len(), 1);
    assert!(!corpus.is_empty());
    assert!(corpus.get(&id).is_some());
}

#[test]
fn test_proof_corpus_find_similar() {
    let mut corpus = ProofCorpus::new();

    for i in 0..5 {
        let result = make_learnable_result(&format!("inv_{}", i), BackendId::Lean4, vec!["decide"]);
        corpus.insert(&result);
    }

    let query = make_invariant("new_inv");
    let similar = corpus.find_similar(&query, 3);

    assert!(similar.len() <= 3);
}

#[test]
fn test_proof_corpus_history() {
    let mut corpus = ProofCorpus::new();

    for i in 0..3 {
        let result = make_learnable_result(&format!("prop_{}", i), BackendId::Lean4, vec!["simp"]);
        corpus.insert(&result);
    }

    let history = corpus.history(TimePeriod::Day);
    assert_eq!(history.total_count, 3);
}

// =============================================================================
// CounterexampleCorpus Tests
// =============================================================================

#[test]
fn test_counterexample_corpus_new() {
    let corpus = CounterexampleCorpus::new();
    assert_eq!(corpus.len(), 0);
    assert!(corpus.is_empty());
}

#[test]
fn test_counterexample_corpus_insert() {
    let mut corpus = CounterexampleCorpus::new();

    let cx = make_counterexample(&[("x", 1)], "error");
    let id = corpus.insert("prop", BackendId::TlaPlus, cx, None);

    assert_eq!(corpus.len(), 1);
    assert!(corpus.get(&id).is_some());
}

#[test]
fn test_counterexample_corpus_find_similar() {
    let mut corpus = CounterexampleCorpus::new();

    for i in 0..5 {
        let cx = make_counterexample(&[("n", i)], "error");
        corpus.insert(&format!("prop_{}", i), BackendId::TlaPlus, cx, None);
    }

    let query = make_counterexample(&[("n", 2)], "error");
    let similar = corpus.find_similar(&query, 3);

    assert!(similar.len() <= 3);
}

// =============================================================================
// Persistence Tests
// =============================================================================

#[test]
fn test_learning_system_save_and_load() {
    let mut system = ProofLearningSystem::new();

    // Add some data
    system.record(&make_learnable_result(
        "persist_test",
        BackendId::Lean4,
        vec!["simp", "decide"],
    ));

    let cx = make_counterexample(&[("x", 42)], "test error");
    system.record_counterexample("cx_prop", BackendId::TlaPlus, cx, None);

    // Save
    let dir = temp_dir("save_load");
    system.save_to_dir(&dir).unwrap();

    // Load
    let loaded = ProofLearningSystem::load_from_dir(&dir).unwrap();

    assert_eq!(loaded.proof_count(), 1);
    assert_eq!(loaded.counterexample_count(), 1);

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn test_load_from_missing_dir() {
    let dir = temp_dir("missing");
    std::fs::remove_dir_all(&dir).ok();

    let loaded = ProofLearningSystem::load_from_dir(&dir).unwrap();
    assert_eq!(loaded.proof_count(), 0);
    assert_eq!(loaded.counterexample_count(), 0);
}

#[test]
fn test_proof_corpus_save_and_load() {
    let mut corpus = ProofCorpus::new();

    corpus.insert(&make_learnable_result(
        "test1",
        BackendId::Lean4,
        vec!["decide"],
    ));
    corpus.insert(&make_learnable_result(
        "test2",
        BackendId::Coq,
        vec!["auto"],
    ));

    let dir = temp_dir("corpus_persist");
    let path = dir.join("corpus.json");
    corpus.save_to_file(&path).unwrap();

    let loaded = ProofCorpus::load_or_default(&path).unwrap();
    assert_eq!(loaded.len(), 2);

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn test_tactic_database_save_and_load() {
    let mut db = TacticDatabase::new();

    let context = TacticContext::from_features(&PropertyFeatures::default());
    for _ in 0..5 {
        db.record_success(&context, "simp");
    }

    let dir = temp_dir("tactics_persist");
    let path = dir.join("tactics.json");
    db.save_to_file(&path).unwrap();

    let loaded = TacticDatabase::load_or_default(&path).unwrap();
    assert_eq!(loaded.total_observations(), 5);

    std::fs::remove_dir_all(dir).ok();
}

// =============================================================================
// Embedding Tests
// =============================================================================

#[test]
fn test_property_embedder() {
    let mut embedder = PropertyEmbedder::new();

    let property = make_theorem("test_theorem");
    let embedding = embedder.embed(&property);

    // Embedding has a vector field
    assert_eq!(embedding.vector.len(), EMBEDDING_DIM);
    // Embeddings should have non-zero values
    assert!(embedding.vector.iter().any(|&x| x != 0.0));
}

#[test]
fn test_embedding_similarity() {
    let mut embedder = PropertyEmbedder::new();

    let prop1 = make_invariant("buffer_safety");
    let prop2 = make_invariant("buffer_check");
    let prop3 = make_invariant("completely_different_name");

    let emb1 = embedder.embed(&prop1);
    let emb2 = embedder.embed(&prop2);
    let emb3 = embedder.embed(&prop3);

    // Compute cosine similarity using the vector fields
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    let sim_similar = cosine_sim(&emb1.vector, &emb2.vector);
    let sim_different = cosine_sim(&emb1.vector, &emb3.vector);

    // Similarity should be in valid range
    assert!((-1.0..=1.0).contains(&sim_similar));
    assert!((-1.0..=1.0).contains(&sim_different));
}

#[test]
fn test_embedding_index_builder() {
    let builder = EmbeddingIndexBuilder::new();
    let index = builder.build();

    assert!(index.is_empty());
}

#[test]
fn test_embedding_index_add_and_search() {
    let mut builder = EmbeddingIndexBuilder::new();

    // Add some properties using the correct API
    for i in 0..10 {
        let prop = make_invariant(&format!("property_{}", i));
        builder.add(format!("id_{}", i), &prop);
    }

    let index = builder.build();
    assert_eq!(index.len(), 10);

    // Search using find_nearest
    let query = make_invariant("property_5");
    let results = index.find_nearest(&query, 3);

    assert!(results.len() <= 3);
    // Results should include similarity scores
    for (id, score) in &results {
        assert!(!id.is_empty());
        assert!(*score >= -1.0 && *score <= 1.0);
    }
}

// =============================================================================
// PropertyFeatures Tests
// =============================================================================

#[test]
fn test_property_features_default() {
    let features = PropertyFeatures::default();
    // PropertyFeatures has function_calls (not function_names) and variable_count
    assert_eq!(features.function_calls, 0);
    assert_eq!(features.variable_count, 0);
}

#[test]
fn test_tactic_context_from_features() {
    let features = PropertyFeatures {
        property_type: "Invariant".to_string(),
        depth: 2,
        quantifier_depth: 1,
        implication_count: 0,
        arithmetic_ops: 0,
        function_calls: 1,
        variable_count: 1,
        has_temporal: false,
        type_refs: vec![],
        keywords: vec!["test".to_string()],
    };

    let context = TacticContext::from_features(&features);

    // Context should be derived from features
    assert!(!context.property_type.is_empty());
}

// =============================================================================
// TimePeriod Tests
// =============================================================================

#[test]
fn test_time_period_variants() {
    // TimePeriod has Day, Week, Month (no Hour)
    let _day = TimePeriod::Day;
    let _week = TimePeriod::Week;
    let _month = TimePeriod::Month;
}

// =============================================================================
// Re-export Tests
// =============================================================================

#[test]
fn test_public_api_exports() {
    // Verify all expected types are exported
    let _ = ProofLearningSystem::new();
    let _ = ProofCorpus::new();
    let _ = CounterexampleCorpus::new();
    let _ = TacticDatabase::new();
    let _ = PropertyEmbedder::new();
    let _ = EmbeddingIndexBuilder::new();
    let _ = PropertyFeatures::default();
    // TacticContext doesn't implement Default, use from_features
    let _ = TacticContext::from_features(&PropertyFeatures::default());
}

// =============================================================================
// Complex Scenario Tests
// =============================================================================

#[test]
fn test_end_to_end_proof_learning() {
    let mut system = ProofLearningSystem::new();

    // Simulate a verification session
    let backends = [BackendId::Lean4, BackendId::Coq, BackendId::Isabelle];
    let tactics_sets = [
        vec!["simp", "decide"],
        vec!["auto", "blast"],
        vec!["simp_all"],
    ];

    for i in 0..15 {
        let backend = backends[i % 3];
        let tactics = &tactics_sets[i % 3];

        let result = make_learnable_result(&format!("theorem_{}", i), backend, tactics.to_vec());
        system.record(&result);
    }

    assert_eq!(system.proof_count(), 15);

    // Test similarity search
    let query = make_theorem("theorem_new");
    let similar = system.find_similar(&query, 5);
    assert!(!similar.is_empty());

    // Test tactic suggestions
    let suggestions = system.suggest_tactics(&query, 5);
    assert!(!suggestions.is_empty());

    // Test history
    let history = system.proof_history(TimePeriod::Day);
    assert_eq!(history.total_count, 15);
}

#[test]
fn test_end_to_end_counterexample_learning() {
    let mut system = ProofLearningSystem::new();

    // Record various counterexamples
    for i in 0..10 {
        let cx = make_counterexample(
            &[("index", i), ("size", 100)],
            &format!("array bounds violation at {}", i),
        );
        system.record_counterexample(
            &format!("bounds_check_{}", i),
            BackendId::Kani,
            cx,
            Some("bounds_error".to_string()),
        );
    }

    assert_eq!(system.counterexample_count(), 10);

    // Test similarity search
    let query = make_counterexample(&[("index", 5), ("size", 100)], "bounds error");
    let similar = system.find_similar_counterexamples(&query, 5);
    assert!(!similar.is_empty());

    // Test keyword search
    let keyword_results = system.search_counterexamples_by_keywords("bounds", 5);
    assert!(!keyword_results.is_empty());
}

#[test]
fn test_mixed_learning_workflow() {
    let mut system = ProofLearningSystem::new();

    // Simulate mixed verification results
    // 1. Some successful proofs
    for i in 0..5 {
        system.record(&make_learnable_result(
            &format!("success_{}", i),
            BackendId::Lean4,
            vec!["simp"],
        ));
    }

    // 2. Some counterexamples
    for i in 0..3 {
        let cx = make_counterexample(&[("n", i)], "property violated");
        system.record_counterexample(&format!("failure_{}", i), BackendId::TlaPlus, cx, None);
    }

    // 3. Some failed attempts (recorded for tactic learning)
    for i in 0..3 {
        let result = LearnableResult {
            property: make_invariant(&format!("timeout_{}", i)),
            backend: BackendId::Lean4,
            status: VerificationStatus::Unknown {
                reason: "timeout".to_string(),
            },
            tactics: vec!["simp".to_string()],
            time_taken: Duration::from_secs(60),
            proof_output: None,
        };
        system.record(&result);
    }

    // Verify corpus state
    assert_eq!(system.proof_count(), 5);
    assert_eq!(system.counterexample_count(), 3);

    // Persist and reload
    let dir = temp_dir("mixed_workflow");
    system.save_to_dir(&dir).unwrap();

    let loaded = ProofLearningSystem::load_from_dir(&dir).unwrap();
    assert_eq!(loaded.proof_count(), 5);
    assert_eq!(loaded.counterexample_count(), 3);

    std::fs::remove_dir_all(dir).ok();
}
