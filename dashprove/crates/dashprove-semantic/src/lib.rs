//! Semantic and fuzzy verification for LLM outputs
//!
//! This crate provides semantic verification capabilities for DashProve,
//! enabling verification of LLM outputs where exact matching is not possible
//! or desirable.
//!
//! # Key Features
//!
//! - **Semantic similarity checking**: Compare text outputs using embeddings
//! - **Statistical property verification**: Verify properties across multiple samples
//! - **Configurable thresholds**: Tune similarity thresholds for your use case
//! - **Embedding backends**: Support for local (TF-IDF, BM25) and API-based embeddings
//!
//! # Architecture
//!
//! The crate is organized into several modules:
//!
//! - [`embedding`]: Text embedding generation (TF-IDF, API-based)
//! - [`similarity`]: Semantic similarity computation
//! - [`statistical`]: Statistical property verification over samples
//! - [`predicate`]: Built-in semantic predicates for common checks
//!
//! # Example
//!
//! ```ignore
//! use dashprove_semantic::{SemanticVerifier, SemanticConfig};
//!
//! let config = SemanticConfig::default();
//! let verifier = SemanticVerifier::new(config);
//!
//! // Check semantic similarity
//! let result = verifier.check_similarity(
//!     "The file was created successfully",
//!     "File creation completed",
//!     0.8,
//! ).await?;
//!
//! assert!(result.passed);
//! ```

pub mod embedding;
pub mod error;
pub mod predicate;
pub mod similarity;
pub mod statistical;

pub use embedding::{
    EmbeddingBackend, EmbeddingConfig, LocalEmbedding, TextEmbedding, TfIdfEmbedder,
};
pub use error::{SemanticError, SemanticResult};
pub use predicate::{
    AddressesQuestion, SemanticPredicate, SemanticPredicateResult, SemanticSimilarity,
};
pub use similarity::{SemanticChecker, SemanticConfig, SimilarityResult};
pub use statistical::{
    SampleResult, StatisticalConfig, StatisticalProperty, StatisticalResult, StatisticalVerifier,
};

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // ==================== Strategies ====================

    fn embedding_config_strategy() -> impl Strategy<Value = EmbeddingConfig> {
        (1usize..1024, any::<bool>(), 1usize..10, 100usize..20000).prop_map(
            |(dimension, normalize, min_term_freq, max_vocab_size)| EmbeddingConfig {
                dimension,
                normalize,
                min_term_freq,
                max_vocab_size,
            },
        )
    }

    fn similarity_metric_strategy() -> impl Strategy<Value = similarity::SimilarityMetric> {
        prop_oneof![
            Just(similarity::SimilarityMetric::Cosine),
            Just(similarity::SimilarityMetric::Euclidean),
            Just(similarity::SimilarityMetric::DotProduct),
        ]
    }

    fn backend_config_strategy() -> impl Strategy<Value = similarity::BackendConfig> {
        prop_oneof![
            (1usize..512).prop_map(|dimension| similarity::BackendConfig::Local { dimension }),
            (100usize..10000)
                .prop_map(|max_vocab_size| similarity::BackendConfig::TfIdf { max_vocab_size }),
        ]
    }

    fn semantic_config_strategy() -> impl Strategy<Value = similarity::SemanticConfig> {
        (
            0.0f64..1.0,
            similarity_metric_strategy(),
            backend_config_strategy(),
            any::<bool>(),
        )
            .prop_map(|(threshold, metric, backend, store_sources)| {
                similarity::SemanticConfig {
                    default_threshold: threshold,
                    metric,
                    backend,
                    store_sources,
                }
            })
    }

    fn statistical_config_strategy() -> impl Strategy<Value = StatisticalConfig> {
        (
            1usize..50,
            50usize..500,
            0.80f64..0.99,
            any::<bool>(),
            100u64..10000,
        )
            .prop_map(
                |(
                    min_samples,
                    max_samples,
                    confidence_level,
                    early_stopping,
                    sample_timeout_ms,
                )| {
                    StatisticalConfig {
                        min_samples,
                        max_samples: max_samples.max(min_samples + 1),
                        confidence_level,
                        early_stopping,
                        sample_timeout_ms,
                    }
                },
            )
    }

    proptest! {
        // ==================== EmbeddingConfig Tests ====================

        #[test]
        fn embedding_config_default_values(_x in 0..1i32) {
            let config = EmbeddingConfig::default();
            prop_assert!(config.dimension > 0);
            prop_assert!(config.min_term_freq >= 1);
            prop_assert!(config.max_vocab_size > 0);
        }

        #[test]
        fn embedding_config_fields_preserved(config in embedding_config_strategy()) {
            prop_assert!(config.dimension > 0);
            prop_assert!(config.min_term_freq >= 1);
            prop_assert!(config.max_vocab_size > 0);
        }

        // ==================== TextEmbedding Tests ====================

        #[test]
        fn text_embedding_new_has_dimension(vec in prop::collection::vec(-10.0f32..10.0, 1..100)) {
            let emb = TextEmbedding::new(vec.clone());
            prop_assert_eq!(emb.dimension(), vec.len());
            prop_assert!(emb.source.is_none());
        }

        #[test]
        fn text_embedding_with_source_preserves_source(
            vec in prop::collection::vec(-10.0f32..10.0, 1..100),
            source in "[a-zA-Z ]{1,50}"
        ) {
            let emb = TextEmbedding::with_source(vec.clone(), source.clone());
            prop_assert_eq!(emb.dimension(), vec.len());
            prop_assert_eq!(emb.source, Some(source));
        }

        #[test]
        fn text_embedding_l2_norm_non_negative(vec in prop::collection::vec(-10.0f32..10.0, 1..100)) {
            let emb = TextEmbedding::new(vec);
            prop_assert!(emb.l2_norm() >= 0.0);
        }

        #[test]
        fn text_embedding_normalized_has_unit_norm(vec in prop::collection::vec(0.1f32..10.0, 1..100)) {
            let emb = TextEmbedding::new(vec);
            let normalized = emb.normalized();
            let norm = normalized.l2_norm();
            prop_assert!((norm - 1.0).abs() < 0.001 || norm < 0.001);
        }

        #[test]
        fn text_embedding_cosine_similarity_range(
            v1 in prop::collection::vec(-10.0f32..10.0, 10..50),
            v2 in prop::collection::vec(-10.0f32..10.0, 10..50)
        ) {
            let dim = v1.len().min(v2.len());
            let emb1 = TextEmbedding::new(v1[..dim].to_vec());
            let emb2 = TextEmbedding::new(v2[..dim].to_vec());
            let sim = emb1.cosine_similarity(&emb2);
            prop_assert!((-1.0..=1.0).contains(&sim));
        }

        #[test]
        fn text_embedding_self_similarity_is_one(vec in prop::collection::vec(0.1f32..10.0, 1..100)) {
            let emb = TextEmbedding::new(vec);
            let sim = emb.cosine_similarity(&emb);
            prop_assert!((sim - 1.0).abs() < 0.001 || emb.l2_norm() < 0.001);
        }

        #[test]
        fn text_embedding_l2_distance_non_negative(
            v1 in prop::collection::vec(-10.0f32..10.0, 10..50),
            v2 in prop::collection::vec(-10.0f32..10.0, 10..50)
        ) {
            let dim = v1.len().min(v2.len());
            let emb1 = TextEmbedding::new(v1[..dim].to_vec());
            let emb2 = TextEmbedding::new(v2[..dim].to_vec());
            let dist = emb1.l2_distance(&emb2);
            prop_assert!(dist >= 0.0);
        }

        #[test]
        fn text_embedding_self_distance_is_zero(vec in prop::collection::vec(-10.0f32..10.0, 1..100)) {
            let emb = TextEmbedding::new(vec);
            let dist = emb.l2_distance(&emb);
            prop_assert!(dist.abs() < 0.001);
        }

        // ==================== TfIdfEmbedder Tests ====================

        #[test]
        fn tfidf_embedder_new_is_empty(config in embedding_config_strategy()) {
            let embedder = TfIdfEmbedder::new(config);
            prop_assert_eq!(embedder.vocab_size(), 0);
        }

        #[test]
        fn tfidf_embedder_fit_builds_vocabulary(_x in 0..1i32) {
            let config = EmbeddingConfig::default();
            let mut embedder = TfIdfEmbedder::new(config);
            let docs = vec!["hello world", "world peace", "hello there"];
            embedder.fit(&docs);
            prop_assert!(embedder.vocab_size() > 0);
            prop_assert!(embedder.vocabulary().contains_key("hello"));
            prop_assert!(embedder.vocabulary().contains_key("world"));
        }

        #[test]
        fn tfidf_embedder_idf_non_empty_after_fit(_x in 0..1i32) {
            let config = EmbeddingConfig::default();
            let mut embedder = TfIdfEmbedder::new(config);
            let docs = vec!["test document"];
            embedder.fit(&docs);
            prop_assert!(!embedder.idf().is_empty());
        }

        // ==================== LocalEmbedding Tests ====================

        #[test]
        fn local_embedding_has_dimension(dim in 1usize..512) {
            let embedder = LocalEmbedding::new(dim);
            prop_assert_eq!(embedder.dimension(), dim);
        }

        #[test]
        fn local_embedding_default_dimension(_x in 0..1i32) {
            let embedder = LocalEmbedding::default();
            prop_assert_eq!(embedder.dimension(), 256);
        }

        #[test]
        fn local_embedding_name_is_local(_x in 0..1i32) {
            let embedder = LocalEmbedding::default();
            prop_assert_eq!(embedder.name(), "local");
        }

        // ==================== SemanticConfig Tests ====================

        #[test]
        fn semantic_config_default_values(_x in 0..1i32) {
            let config = similarity::SemanticConfig::default();
            prop_assert!(config.default_threshold > 0.0 && config.default_threshold <= 1.0);
            prop_assert!(config.store_sources);
        }

        #[test]
        fn semantic_config_fields_preserved(config in semantic_config_strategy()) {
            prop_assert!(config.default_threshold >= 0.0 && config.default_threshold <= 1.0);
        }

        // ==================== SimilarityResult Tests ====================

        #[test]
        fn similarity_result_pass_has_passed_true(sim in 0.0f64..1.0, threshold in 0.0f64..1.0) {
            let result = similarity::SimilarityResult::pass(sim, threshold);
            prop_assert!(result.passed);
            prop_assert_eq!(result.similarity, sim);
            prop_assert_eq!(result.threshold, threshold);
        }

        #[test]
        fn similarity_result_fail_has_passed_false(sim in 0.0f64..1.0, threshold in 0.0f64..1.0) {
            let result = similarity::SimilarityResult::fail(sim, threshold);
            prop_assert!(!result.passed);
            prop_assert_eq!(result.similarity, sim);
            prop_assert_eq!(result.threshold, threshold);
        }

        #[test]
        fn similarity_result_with_source_preserves(sim in 0.0f64..1.0, source in "[a-zA-Z ]{1,30}") {
            let result = similarity::SimilarityResult::pass(sim, 0.5).with_source(source.clone());
            prop_assert_eq!(result.source, Some(source));
        }

        #[test]
        fn similarity_result_with_target_preserves(sim in 0.0f64..1.0, target in "[a-zA-Z ]{1,30}") {
            let result = similarity::SimilarityResult::pass(sim, 0.5).with_target(target.clone());
            prop_assert_eq!(result.target, Some(target));
        }

        #[test]
        fn similarity_result_with_details_preserves(sim in 0.0f64..1.0, details in "[a-zA-Z ]{1,30}") {
            let result = similarity::SimilarityResult::pass(sim, 0.5).with_details(details.clone());
            prop_assert_eq!(result.details, Some(details));
        }

        // ==================== StatisticalConfig Tests ====================

        #[test]
        fn statistical_config_default_values(_x in 0..1i32) {
            let config = StatisticalConfig::default();
            prop_assert!(config.min_samples > 0);
            prop_assert!(config.max_samples > config.min_samples);
            prop_assert!(config.confidence_level > 0.0 && config.confidence_level < 1.0);
        }

        #[test]
        fn statistical_config_fields_preserved(config in statistical_config_strategy()) {
            prop_assert!(config.min_samples > 0);
            prop_assert!(config.max_samples >= config.min_samples);
            prop_assert!(config.confidence_level > 0.0 && config.confidence_level < 1.0);
        }

        // ==================== SampleResult Tests ====================

        #[test]
        fn sample_result_fields_preserved(
            idx in 0usize..1000,
            value in "[a-zA-Z ]{1,30}",
            passed in any::<bool>(),
            score in 0.0f64..1.0
        ) {
            let result = SampleResult {
                index: idx,
                value: value.clone(),
                passed,
                score,
                details: None,
            };
            prop_assert_eq!(result.index, idx);
            prop_assert_eq!(result.value, value);
            prop_assert_eq!(result.passed, passed);
            prop_assert_eq!(result.score, score);
        }

        // ==================== StatisticalResult Tests ====================

        #[test]
        fn statistical_result_is_significant_above(ci_lower in 0.0f64..1.0, threshold in 0.0f64..1.0) {
            let result = StatisticalResult {
                passed: true,
                num_samples: 100,
                num_passed: 90,
                success_rate: 0.9,
                ci_lower,
                ci_upper: 1.0,
                p_value: None,
                samples: vec![],
                early_stopped: false,
                explanation: String::new(),
            };
            prop_assert_eq!(result.is_significant_above(threshold), ci_lower >= threshold);
        }

        #[test]
        fn statistical_result_is_significant_below(ci_upper in 0.0f64..1.0, threshold in 0.0f64..1.0) {
            let result = StatisticalResult {
                passed: true,
                num_samples: 100,
                num_passed: 90,
                success_rate: 0.9,
                ci_lower: 0.0,
                ci_upper,
                p_value: None,
                samples: vec![],
                early_stopped: false,
                explanation: String::new(),
            };
            prop_assert_eq!(result.is_significant_below(threshold), ci_upper < threshold);
        }

        // ==================== StatisticalVerifier Tests ====================

        #[test]
        fn statistical_verifier_new_creates_instance(config in statistical_config_strategy()) {
            let _verifier = StatisticalVerifier::new(config);
            prop_assert!(true);
        }

        #[test]
        fn statistical_verifier_with_defaults_creates_instance(_x in 0..1i32) {
            let _verifier = StatisticalVerifier::with_defaults();
            prop_assert!(true);
        }

        // ==================== SemanticPredicateResult Tests ====================

        #[test]
        fn predicate_result_pass_has_passed_true(
            predicate in "[a-z]{1,20}",
            confidence in 0.0f64..1.0,
            explanation in "[a-zA-Z ]{1,50}"
        ) {
            let result = predicate::SemanticPredicateResult::pass(&predicate, confidence, &explanation);
            prop_assert!(result.passed);
            prop_assert_eq!(result.predicate, predicate);
            prop_assert_eq!(result.confidence, confidence);
            prop_assert_eq!(result.explanation, explanation);
        }

        #[test]
        fn predicate_result_fail_has_passed_false(
            predicate in "[a-z]{1,20}",
            confidence in 0.0f64..1.0,
            explanation in "[a-zA-Z ]{1,50}"
        ) {
            let result = predicate::SemanticPredicateResult::fail(&predicate, confidence, &explanation);
            prop_assert!(!result.passed);
            prop_assert_eq!(result.predicate, predicate);
            prop_assert_eq!(result.confidence, confidence);
        }

        #[test]
        fn predicate_result_with_metadata_preserves(_x in 0..1i32) {
            let metadata = serde_json::json!({"key": "value"});
            let result = predicate::SemanticPredicateResult::pass("test", 0.9, "explanation")
                .with_metadata(metadata.clone());
            prop_assert!(result.metadata.is_some());
            prop_assert_eq!(&result.metadata.unwrap()["key"], "value");
        }

        // ==================== PredicateInputs Tests ====================

        #[test]
        fn predicate_inputs_primary_preserves(text in "[a-zA-Z ]{1,50}") {
            let inputs = predicate::PredicateInputs::primary(&text);
            prop_assert_eq!(inputs.primary, text);
            prop_assert!(inputs.secondary.is_none());
            prop_assert!(inputs.references.is_empty());
        }

        #[test]
        fn predicate_inputs_pair_preserves(
            primary in "[a-zA-Z ]{1,30}",
            secondary in "[a-zA-Z ]{1,30}"
        ) {
            let inputs = predicate::PredicateInputs::pair(&primary, &secondary);
            prop_assert_eq!(inputs.primary, primary);
            prop_assert_eq!(inputs.secondary, Some(secondary));
        }

        #[test]
        fn predicate_inputs_with_references_preserves(
            text in "[a-zA-Z ]{1,30}",
            refs in prop::collection::vec("[a-zA-Z ]{1,20}", 1..5)
        ) {
            let inputs = predicate::PredicateInputs::primary(&text).with_references(refs.clone());
            prop_assert_eq!(inputs.references.len(), refs.len());
        }

        // ==================== SemanticSimilarity Tests ====================

        #[test]
        fn semantic_similarity_threshold_preserved(threshold in 0.0f64..1.0) {
            let pred = SemanticSimilarity::new(threshold);
            prop_assert_eq!(pred.threshold(), threshold);
        }

        #[test]
        fn semantic_similarity_name_is_semantic_similarity(_x in 0..1i32) {
            let pred = SemanticSimilarity::new(0.5);
            prop_assert_eq!(pred.name(), "semantic_similarity");
        }

        // ==================== AddressesQuestion Tests ====================

        #[test]
        fn addresses_question_name_is_addresses_question(_x in 0..1i32) {
            let pred = AddressesQuestion::new(0.5);
            prop_assert_eq!(pred.name(), "addresses_question");
        }

        // ==================== PredicateRegistry Tests ====================

        #[test]
        fn predicate_registry_new_is_empty(_x in 0..1i32) {
            let registry = predicate::PredicateRegistry::new();
            prop_assert!(registry.list().is_empty());
        }

        #[test]
        fn predicate_registry_with_defaults_has_predicates(_x in 0..1i32) {
            let registry = predicate::PredicateRegistry::with_defaults();
            let names = registry.list();
            prop_assert!(!names.is_empty());
            prop_assert!(registry.get("semantic_similarity").is_some());
            prop_assert!(registry.get("addresses_question").is_some());
        }

        #[test]
        fn predicate_registry_register_adds_predicate(_x in 0..1i32) {
            let mut registry = predicate::PredicateRegistry::new();
            let pred = std::sync::Arc::new(SemanticSimilarity::new(0.5));
            registry.register(pred);
            prop_assert!(registry.get("semantic_similarity").is_some());
        }

        // ==================== SemanticError Tests ====================

        #[test]
        fn semantic_error_embedding_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = SemanticError::Embedding(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        #[test]
        fn semantic_error_similarity_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = SemanticError::Similarity(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        #[test]
        fn semantic_error_statistical_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = SemanticError::Statistical(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        #[test]
        fn semantic_error_config_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = SemanticError::Config(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        #[test]
        fn semantic_error_sample_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = SemanticError::Sample(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        #[test]
        fn semantic_error_predicate_preserves_message(msg in "[a-zA-Z ]{1,30}") {
            let err = SemanticError::Predicate(msg.clone());
            let display = format!("{}", err);
            prop_assert!(display.contains(&msg));
        }

        // ==================== SemanticCheckerBuilder Tests ====================

        #[test]
        fn semantic_checker_builder_threshold_preserved(threshold in 0.0f64..1.0) {
            let checker = similarity::SemanticCheckerBuilder::new()
                .threshold(threshold)
                .build();
            prop_assert_eq!(checker.config().default_threshold, threshold);
        }

        #[test]
        fn semantic_checker_builder_metric_preserved(metric in similarity_metric_strategy()) {
            let checker = similarity::SemanticCheckerBuilder::new()
                .metric(metric)
                .build();
            prop_assert_eq!(checker.config().metric, metric);
        }

        #[test]
        fn semantic_checker_builder_store_sources_preserved(store in any::<bool>()) {
            let checker = similarity::SemanticCheckerBuilder::new()
                .store_sources(store)
                .build();
            prop_assert_eq!(checker.config().store_sources, store);
        }

        #[test]
        fn semantic_checker_builder_local_embeddings_sets_dimension(dim in 1usize..512) {
            let checker = similarity::SemanticCheckerBuilder::new()
                .local_embeddings(dim)
                .build();
            if let similarity::BackendConfig::Local { dimension } = checker.config().backend {
                prop_assert_eq!(dimension, dim);
            } else {
                prop_assert!(false, "Expected Local backend");
            }
        }
    }
}
