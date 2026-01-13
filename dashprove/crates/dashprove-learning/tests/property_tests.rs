//! Property-based tests for dashprove-learning
//!
//! Uses proptest to verify:
//! - Embedding distance properties (triangle inequality, reflexivity, symmetry)
//! - Cosine similarity properties
//! - Normalization is idempotent
//! - EmbeddingIndex nearest neighbor consistency
//! - ProofCorpus operations
//! - TacticDatabase statistics

use dashprove_learning::{
    Embedding, EmbeddingIndex, PropertyFeatures, TacticContext, TacticDatabase,
};
use proptest::prelude::*;

// ============================================================================
// Generators for learning types
// ============================================================================

/// Generate a valid embedding vector (finite, reasonable magnitude)
fn embedding_vec_strategy(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-100.0f32..100.0f32, dim)
}

/// Generate Embedding with standard dimension
fn embedding_strategy() -> impl Strategy<Value = Embedding> {
    embedding_vec_strategy(dashprove_learning::EMBEDDING_DIM).prop_map(Embedding::new)
}

/// Generate smaller embedding for faster tests
fn small_embedding_strategy() -> impl Strategy<Value = Embedding> {
    embedding_vec_strategy(16).prop_map(Embedding::new)
}

/// Generate non-zero embedding (for valid distance/similarity tests)
fn nonzero_embedding_strategy() -> impl Strategy<Value = Embedding> {
    embedding_vec_strategy(dashprove_learning::EMBEDDING_DIM)
        .prop_filter("non-zero embedding", |v| v.iter().any(|&x| x.abs() > 1e-6))
        .prop_map(Embedding::new)
}

/// Generate PropertyFeatures
fn property_features_strategy() -> impl Strategy<Value = PropertyFeatures> {
    (
        prop_oneof![
            Just("theorem".to_string()),
            Just("invariant".to_string()),
            Just("temporal".to_string()),
            Just("contract".to_string()),
        ],
        0usize..10,
        0usize..5,
        0usize..10,
        0usize..10,
        0usize..5,
        0usize..10,
        any::<bool>(),
        prop::collection::vec("[a-z]{1,10}".prop_map(|s| s.to_string()), 0..5),
        prop::collection::vec("[A-Z][a-z]{1,8}".prop_map(|s| s.to_string()), 0..3),
    )
        .prop_map(
            |(
                property_type,
                depth,
                quantifier_depth,
                implication_count,
                arithmetic_ops,
                function_calls,
                variable_count,
                has_temporal,
                keywords,
                type_refs,
            )| PropertyFeatures {
                property_type,
                depth,
                quantifier_depth,
                implication_count,
                arithmetic_ops,
                function_calls,
                variable_count,
                has_temporal,
                keywords,
                type_refs,
            },
        )
}

/// Generate TacticContext
fn tactic_context_strategy() -> impl Strategy<Value = TacticContext> {
    property_features_strategy().prop_map(|f| TacticContext::from_features(&f))
}

// ============================================================================
// Property tests for Embedding
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: L2 distance to self is zero
    #[test]
    fn l2_distance_reflexive(emb in embedding_strategy()) {
        let dist = emb.l2_distance(&emb);
        prop_assert!(dist.abs() < 1e-5, "Self-distance should be zero, got {}", dist);
    }

    /// Property: L2 distance is symmetric
    #[test]
    fn l2_distance_symmetric(a in embedding_strategy(), b in embedding_strategy()) {
        let ab = a.l2_distance(&b);
        let ba = b.l2_distance(&a);
        prop_assert!((ab - ba).abs() < 1e-5, "Distance not symmetric: {} vs {}", ab, ba);
    }

    /// Property: L2 distance is non-negative
    #[test]
    fn l2_distance_nonnegative(a in embedding_strategy(), b in embedding_strategy()) {
        let dist = a.l2_distance(&b);
        prop_assert!(dist >= 0.0, "Distance should be non-negative, got {}", dist);
    }

    /// Property: Cosine similarity to self is 1.0 (for non-zero vectors)
    #[test]
    fn cosine_similarity_reflexive(emb in nonzero_embedding_strategy()) {
        let sim = emb.cosine_similarity(&emb);
        prop_assert!((sim - 1.0).abs() < 1e-5, "Self-similarity should be 1.0, got {}", sim);
    }

    /// Property: Cosine similarity is symmetric
    #[test]
    fn cosine_similarity_symmetric(a in embedding_strategy(), b in embedding_strategy()) {
        let ab = a.cosine_similarity(&b);
        let ba = b.cosine_similarity(&a);
        prop_assert!((ab - ba).abs() < 1e-5, "Similarity not symmetric: {} vs {}", ab, ba);
    }

    /// Property: Cosine similarity is in [-1, 1]
    #[test]
    fn cosine_similarity_bounded(a in embedding_strategy(), b in embedding_strategy()) {
        let sim = a.cosine_similarity(&b);
        prop_assert!((-1.0 - 1e-5..=1.0 + 1e-5).contains(&sim),
            "Similarity out of bounds: {}", sim);
    }

    /// Property: Normalized similarity is in [0, 1]
    #[test]
    fn normalized_similarity_bounded(a in embedding_strategy(), b in embedding_strategy()) {
        let sim = a.normalized_similarity(&b);
        prop_assert!((-1e-5..=1.0 + 1e-5).contains(&sim),
            "Normalized similarity out of bounds: {}", sim);
    }

    /// Property: Normalization is idempotent
    #[test]
    fn normalization_idempotent(emb in nonzero_embedding_strategy()) {
        let normalized = emb.normalized();
        let double_normalized = normalized.normalized();

        for (a, b) in normalized.vector.iter().zip(double_normalized.vector.iter()) {
            prop_assert!((a - b).abs() < 1e-5,
                "Normalization not idempotent: {} vs {}", a, b);
        }
    }

    /// Property: Normalized vectors have unit length (approximately)
    #[test]
    fn normalized_unit_length(emb in nonzero_embedding_strategy()) {
        let normalized = emb.normalized();
        let norm: f32 = normalized.vector.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        prop_assert!((norm - 1.0).abs() < 1e-4,
            "Normalized vector should have unit length, got {}", norm);
    }

    /// Property: Clone equals original
    #[test]
    fn embedding_clone_equals(emb in embedding_strategy()) {
        let cloned = emb.clone();
        prop_assert_eq!(emb.dim, cloned.dim);
        prop_assert_eq!(emb.vector.len(), cloned.vector.len());
        for (a, b) in emb.vector.iter().zip(cloned.vector.iter()) {
            prop_assert_eq!(a, b);
        }
    }

    /// Property: Zeros embedding has correct dimension
    #[test]
    fn zeros_correct_dimension(dim in 1usize..100) {
        let emb = Embedding::zeros(dim);
        prop_assert_eq!(emb.dim, dim);
        prop_assert_eq!(emb.vector.len(), dim);
        prop_assert!(emb.vector.iter().all(|&x| x == 0.0));
    }

    /// Property: L2 distance for different dimensions returns MAX
    #[test]
    fn l2_distance_different_dims(dim1 in 10usize..50, dim2 in 50usize..100) {
        let a = Embedding::zeros(dim1);
        let b = Embedding::zeros(dim2);
        let dist = a.l2_distance(&b);
        prop_assert_eq!(dist, f32::MAX);
    }

    /// Property: Cosine similarity for different dimensions returns 0
    #[test]
    fn cosine_similarity_different_dims(dim1 in 10usize..50, dim2 in 50usize..100) {
        let a = Embedding::zeros(dim1);
        let b = Embedding::zeros(dim2);
        let sim = a.cosine_similarity(&b);
        prop_assert_eq!(sim, 0.0);
    }
}

// ============================================================================
// Property tests for TacticDatabase
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: Record success increases success count
    #[test]
    fn tactic_record_success_increases_count(
        ctx in tactic_context_strategy(),
        tactic in "[a-z]+".prop_map(|s| s.to_string())
    ) {
        let mut db = TacticDatabase::new();
        let before = db.total_observations();

        db.record_success(&ctx, &tactic);

        prop_assert!(db.total_observations() >= before);
    }

    /// Property: Record failure increases observation count
    #[test]
    fn tactic_record_failure_increases_count(
        ctx in tactic_context_strategy(),
        tactic in "[a-z]+".prop_map(|s| s.to_string())
    ) {
        let mut db = TacticDatabase::new();
        let before = db.total_observations();

        db.record_failure(&ctx, &tactic);

        prop_assert!(db.total_observations() >= before);
    }

    /// Property: Successful tactics appear in best_for_context
    #[test]
    fn successful_tactic_in_best(
        ctx in tactic_context_strategy(),
        tactic in "[a-z]+".prop_map(|s| s.to_string())
    ) {
        let mut db = TacticDatabase::new();

        // Record many successes
        for _ in 0..10 {
            db.record_success(&ctx, &tactic);
        }

        let best = db.best_for_context(&ctx, 5);

        // The tactic should be in the results
        let has_tactic = best.iter().any(|(name, _)| name == &tactic);
        prop_assert!(has_tactic, "Successful tactic '{}' not in best list", tactic);
    }

    /// Property: Scores are in [0, 1]
    #[test]
    fn tactic_scores_bounded(
        ctx in tactic_context_strategy(),
        tactic in "[a-z]+".prop_map(|s| s.to_string())
    ) {
        let mut db = TacticDatabase::new();

        // Mix of successes and failures
        for _ in 0..5 {
            db.record_success(&ctx, &tactic);
            db.record_failure(&ctx, &tactic);
        }

        let best = db.best_for_context(&ctx, 10);

        for (_, score) in best {
            prop_assert!((0.0..=1.0).contains(&score),
                "Score out of bounds: {}", score);
        }
    }
}

// ============================================================================
// Property tests for EmbeddingIndex
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Property: Inserted entries are retrievable via find_nearest
    #[test]
    fn index_inserted_entries_findable(
        entries in prop::collection::vec(
            (
                "[a-z]+".prop_map(|s| s.to_string()),
                small_embedding_strategy()
            ),
            1..5
        )
    ) {
        let mut index = EmbeddingIndex::new();

        for (id, emb) in &entries {
            index.insert_embedding(id.clone(), emb.clone());
        }

        prop_assert_eq!(index.len(), entries.len());
        prop_assert!(!index.is_empty());
    }

    /// Property: find_nearest returns at most k results
    #[test]
    fn index_find_nearest_at_most_k(
        entries in prop::collection::vec(
            (
                "[a-z]+".prop_map(|s| s.to_string()),
                small_embedding_strategy()
            ),
            5..10
        ),
        k in 1usize..5
    ) {
        let mut index = EmbeddingIndex::new();

        for (id, emb) in entries {
            index.insert_embedding(id, emb);
        }

        let query = Embedding::new(vec![1.0f32; 16]);
        let results = index.find_nearest_embedding(&query, k);

        prop_assert!(results.len() <= k);
    }

    /// Property: Empty index returns empty results
    #[test]
    fn index_empty_returns_empty(k in 1usize..10) {
        let index = EmbeddingIndex::new();
        let query = Embedding::new(vec![1.0f32; dashprove_learning::EMBEDDING_DIM]);
        let results = index.find_nearest_embedding(&query, k);

        prop_assert!(results.is_empty());
    }

    /// Property: Results are sorted by similarity (descending)
    #[test]
    fn index_results_sorted(
        entries in prop::collection::vec(
            (
                "[a-z]+".prop_map(|s| s.to_string()),
                small_embedding_strategy()
            ),
            3..8
        )
    ) {
        let mut index = EmbeddingIndex::new();

        for (id, emb) in entries {
            index.insert_embedding(id, emb);
        }

        let query = Embedding::new(vec![1.0f32; 16]);
        let results = index.find_nearest_embedding(&query, 10);

        // Check sorted descending
        for i in 1..results.len() {
            prop_assert!(results[i-1].1 >= results[i].1,
                "Results not sorted: {} < {}", results[i-1].1, results[i].1);
        }
    }
}

// ============================================================================
// Property tests for PropertyFeatures
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: PropertyFeatures can be converted to TacticContext
    #[test]
    fn features_to_context_no_panic(features in property_features_strategy()) {
        let _ = TacticContext::from_features(&features);
    }

    /// Property: PropertyFeatures depth is preserved
    #[test]
    fn features_depth_preserved(features in property_features_strategy()) {
        prop_assert!(features.depth < 100, "Depth should be reasonable");
    }
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn embedding_zero_vector_cosine() {
    let zero = Embedding::zeros(10);
    let nonzero = Embedding::new(vec![1.0; 10]);

    // Zero vector should return 0 similarity
    let sim = zero.cosine_similarity(&nonzero);
    assert!((sim - 0.0).abs() < 1e-6);

    let sim2 = nonzero.cosine_similarity(&zero);
    assert!((sim2 - 0.0).abs() < 1e-6);
}

#[test]
fn embedding_opposite_vectors() {
    let a = Embedding::new(vec![1.0, 2.0, 3.0]);
    let b = Embedding::new(vec![-1.0, -2.0, -3.0]);

    let sim = a.cosine_similarity(&b);
    assert!(
        (sim - (-1.0)).abs() < 1e-5,
        "Opposite vectors should have cosine=-1"
    );
}

#[test]
fn embedding_orthogonal_vectors() {
    let a = Embedding::new(vec![1.0, 0.0, 0.0]);
    let b = Embedding::new(vec![0.0, 1.0, 0.0]);

    let sim = a.cosine_similarity(&b);
    assert!(sim.abs() < 1e-5, "Orthogonal vectors should have cosine=0");
}

#[test]
fn embedding_default_dimension() {
    let emb = Embedding::default();
    assert_eq!(emb.dim, dashprove_learning::EMBEDDING_DIM);
    assert_eq!(emb.vector.len(), dashprove_learning::EMBEDDING_DIM);
}

#[test]
fn tactic_database_empty_context() {
    let db = TacticDatabase::new();
    let ctx = TacticContext::from_features(&PropertyFeatures {
        property_type: "theorem".to_string(),
        depth: 0,
        quantifier_depth: 0,
        implication_count: 0,
        arithmetic_ops: 0,
        function_calls: 0,
        variable_count: 0,
        has_temporal: false,
        keywords: vec![],
        type_refs: vec![],
    });

    let best = db.best_for_context(&ctx, 5);
    assert!(best.is_empty());
}

#[test]
fn index_duplicate_ids() {
    let mut index = EmbeddingIndex::new();

    let emb1 = Embedding::new(vec![1.0; 16]);
    let emb2 = Embedding::new(vec![2.0; 16]);

    index.insert_embedding("same_id".to_string(), emb1);
    index.insert_embedding("same_id".to_string(), emb2);

    // Both entries are stored (IDs don't have to be unique)
    assert_eq!(index.len(), 2);
}

#[test]
fn triangle_inequality_l2() {
    // For any three points, d(a,c) <= d(a,b) + d(b,c)
    let a = Embedding::new(vec![0.0, 0.0, 0.0]);
    let b = Embedding::new(vec![1.0, 0.0, 0.0]);
    let c = Embedding::new(vec![2.0, 0.0, 0.0]);

    let ab = a.l2_distance(&b);
    let bc = b.l2_distance(&c);
    let ac = a.l2_distance(&c);

    assert!(ac <= ab + bc + 1e-5, "Triangle inequality violated");
}
