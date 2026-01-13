//! Tests for expert modules

use super::*;
use crate::embedding::{Embedder, EmbeddingModel};
use crate::store::KnowledgeStore;
use dashprove_backends::BackendId;
use std::path::PathBuf;

#[test]
fn test_property_type_backends() {
    let safety = PropertyType::Safety;
    let backends = safety.relevant_backends();
    assert!(!backends.is_empty());
    assert!(backends.contains(&BackendId::Kani));
}

#[test]
fn test_property_type_description() {
    let safety = PropertyType::Safety;
    assert!(!safety.description().is_empty());
}

#[test]
fn test_expert_context_default() {
    let ctx = ExpertContext::default();
    assert!(ctx.specification.is_none());
    assert!(ctx.property_types.is_empty());
}

#[test]
fn test_prefer_when() {
    let store = KnowledgeStore::new(PathBuf::from("/tmp/test"), 384);
    let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
    let expert = BackendSelectionExpert::new(&store, &embedder);

    let reason = expert.get_prefer_when(BackendId::Kani);
    assert!(reason.contains("Rust"));
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_property_type() -> impl Strategy<Value = PropertyType> {
        prop_oneof![
            Just(PropertyType::Safety),
            Just(PropertyType::Liveness),
            Just(PropertyType::Temporal),
            Just(PropertyType::Correctness),
            Just(PropertyType::Probabilistic),
            Just(PropertyType::NeuralNetwork),
            Just(PropertyType::SecurityProtocol),
            Just(PropertyType::Refinement),
            Just(PropertyType::Smt),
        ]
    }

    fn arb_backend() -> impl Strategy<Value = BackendId> {
        prop_oneof![
            Just(BackendId::Lean4),
            Just(BackendId::TlaPlus),
            Just(BackendId::Kani),
            Just(BackendId::Alloy),
            Just(BackendId::Coq),
            Just(BackendId::Isabelle),
            Just(BackendId::Dafny),
            Just(BackendId::Z3),
            Just(BackendId::Cvc5),
        ]
    }

    proptest! {
        /// Every property type has at least one relevant backend
        #[test]
        fn test_property_types_have_backends(prop_type in arb_property_type()) {
            let backends = prop_type.relevant_backends();
            prop_assert!(!backends.is_empty(),
                "Property type {:?} should have at least one relevant backend", prop_type);
        }

        /// Property type descriptions are non-empty
        #[test]
        fn test_property_descriptions_non_empty(prop_type in arb_property_type()) {
            let desc = prop_type.description();
            prop_assert!(!desc.is_empty(),
                "Description for {:?} should not be empty", prop_type);
        }

        /// Backend prefer_when returns non-empty string
        #[test]
        fn test_prefer_when_non_empty(backend in arb_backend()) {
            let store = KnowledgeStore::new(PathBuf::from("/tmp/test"), 384);
            let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
            let expert = BackendSelectionExpert::new(&store, &embedder);

            let reason = expert.get_prefer_when(backend);
            prop_assert!(!reason.is_empty(),
                "Prefer when for {:?} should not be empty", backend);
        }

        /// Error analysis produces non-empty explanation
        #[test]
        fn test_error_analysis_non_empty(
            error in "[a-zA-Z0-9 ]{10,200}",
            backend in proptest::option::of(arb_backend())
        ) {
            let store = KnowledgeStore::new(PathBuf::from("/tmp/test"), 384);
            let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
            let expert = ErrorExplanationExpert::new(&store, &embedder);

            let (explanation, root_cause) = expert.analyze_error(&error, backend);
            prop_assert!(!explanation.is_empty(), "Explanation should not be empty");
            prop_assert!(!root_cause.is_empty(), "Root cause should not be empty");
        }

        /// Generate fixes always returns at least one fix
        #[test]
        fn test_generate_fixes_non_empty(
            error in "[a-zA-Z0-9 ]{10,200}",
            backend in proptest::option::of(arb_backend())
        ) {
            let store = KnowledgeStore::new(PathBuf::from("/tmp/test"), 384);
            let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
            let expert = ErrorExplanationExpert::new(&store, &embedder);

            let fixes = expert.generate_fixes(&error, backend);
            prop_assert!(!fixes.is_empty(),
                "Should always generate at least one fix suggestion");
        }

        /// Tactic suggestions for theorem provers are non-empty
        #[test]
        fn test_tactic_suggestions_theorem_provers(
            goal in "[a-zA-Z0-9 =∀→]{10,100}",
            backend in prop_oneof![
                Just(BackendId::Lean4),
                Just(BackendId::Coq),
                Just(BackendId::Isabelle),
            ]
        ) {
            let store = KnowledgeStore::new(PathBuf::from("/tmp/test"), 384);
            let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
            let expert = TacticSuggestionExpert::new(&store, &embedder);

            let suggestions = expert.generate_suggestions(&goal, backend);
            // Theorem provers should always have tactic suggestions
            prop_assert!(!suggestions.is_empty() || goal.is_empty(),
                "Theorem prover {:?} should have tactic suggestions", backend);
        }

        /// Compilation steps are non-empty for common backends
        #[test]
        fn test_compilation_steps_non_empty(
            spec in "[a-zA-Z0-9 ]{10,200}",
            backend in prop_oneof![
                Just(BackendId::Lean4),
                Just(BackendId::TlaPlus),
                Just(BackendId::Kani),
            ]
        ) {
            let store = KnowledgeStore::new(PathBuf::from("/tmp/test"), 384);
            let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
            let expert = CompilationGuidanceExpert::new(&store, &embedder);

            let steps = expert.generate_steps(&spec, backend);
            prop_assert!(!steps.is_empty(),
                "Compilation steps for {:?} should not be empty", backend);
        }

        /// Pitfalls list is non-empty for main backends
        #[test]
        fn test_pitfalls_non_empty(
            backend in prop_oneof![
                Just(BackendId::Lean4),
                Just(BackendId::TlaPlus),
                Just(BackendId::Kani),
            ]
        ) {
            let store = KnowledgeStore::new(PathBuf::from("/tmp/test"), 384);
            let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
            let expert = CompilationGuidanceExpert::new(&store, &embedder);

            let pitfalls = expert.get_pitfalls(backend);
            prop_assert!(!pitfalls.is_empty(),
                "Pitfalls for {:?} should not be empty", backend);
        }

        /// Best practices list is non-empty for main backends
        #[test]
        fn test_best_practices_non_empty(
            backend in prop_oneof![
                Just(BackendId::Lean4),
                Just(BackendId::TlaPlus),
                Just(BackendId::Kani),
            ]
        ) {
            let store = KnowledgeStore::new(PathBuf::from("/tmp/test"), 384);
            let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
            let expert = CompilationGuidanceExpert::new(&store, &embedder);

            let practices = expert.get_best_practices(backend);
            prop_assert!(!practices.is_empty(),
                "Best practices for {:?} should not be empty", backend);
        }

        /// ExpertContext can be built with any combination of fields
        #[test]
        fn test_expert_context_builder(
            has_spec in any::<bool>(),
            has_backend in any::<bool>(),
            num_props in 0usize..5,
            num_tags in 0usize..5,
        ) {
            let mut ctx = ExpertContext::default();

            if has_spec {
                ctx.specification = Some("test spec".to_string());
            }
            if has_backend {
                ctx.current_backend = Some(BackendId::Z3);
            }
            for i in 0..num_props {
                ctx.property_types.push(if i % 2 == 0 {
                    PropertyType::Safety
                } else {
                    PropertyType::Liveness
                });
            }
            for i in 0..num_tags {
                ctx.tags.push(format!("tag{}", i));
            }

            prop_assert_eq!(ctx.specification.is_some(), has_spec);
            prop_assert_eq!(ctx.current_backend.is_some(), has_backend);
            prop_assert_eq!(ctx.property_types.len(), num_props);
            prop_assert_eq!(ctx.tags.len(), num_tags);
        }

        /// Confidence values in suggestions are bounded 0-1
        #[test]
        fn test_confidence_bounded(
            goal in "[a-zA-Z0-9 ]{10,50}",
            backend in arb_backend()
        ) {
            let store = KnowledgeStore::new(PathBuf::from("/tmp/test"), 384);
            let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
            let expert = TacticSuggestionExpert::new(&store, &embedder);

            let suggestions = expert.generate_suggestions(&goal, backend);
            for suggestion in suggestions {
                prop_assert!(suggestion.confidence >= 0.0 && suggestion.confidence <= 1.0,
                    "Confidence {} should be in [0, 1]", suggestion.confidence);
            }
        }

        /// Fix confidence values are bounded 0-1
        #[test]
        fn test_fix_confidence_bounded(
            error in "[a-zA-Z0-9 ]{10,100}",
            backend in proptest::option::of(arb_backend())
        ) {
            let store = KnowledgeStore::new(PathBuf::from("/tmp/test"), 384);
            let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);
            let expert = ErrorExplanationExpert::new(&store, &embedder);

            let fixes = expert.generate_fixes(&error, backend);
            for fix in fixes {
                prop_assert!(fix.confidence >= 0.0 && fix.confidence <= 1.0,
                    "Fix confidence {} should be in [0, 1]", fix.confidence);
            }
        }
    }
}
