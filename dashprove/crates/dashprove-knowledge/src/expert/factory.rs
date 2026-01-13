//! Expert factory for unified expert instantiation
//!
//! Provides a single entry point for creating all expert modules with
//! shared dependencies (KnowledgeStore, Embedder, ToolKnowledgeStore).

use crate::embedding::Embedder;
use crate::store::KnowledgeStore;
use crate::tool_knowledge::ToolKnowledgeStore;

use super::{
    BackendSelectionExpert, CompilationGuidanceExpert, ErrorExplanationExpert,
    ResearchRecommendationExpert, TacticSuggestionExpert,
};

/// Factory for creating expert modules with shared dependencies
///
/// The factory holds references to shared resources and provides
/// methods to create each expert type. This ensures all experts
/// share the same ToolKnowledgeStore instance.
///
/// # Example
///
/// ```ignore
/// use dashprove_knowledge::{Embedder, KnowledgeStore, ToolKnowledgeStore};
/// use dashprove_knowledge::expert::ExpertFactory;
///
/// // Create shared dependencies
/// let store = KnowledgeStore::new();
/// let embedder = Embedder::mock();
/// let tool_store = ToolKnowledgeStore::new();
///
/// // Create factory
/// let factory = ExpertFactory::new(&store, &embedder, Some(&tool_store));
///
/// // Create experts as needed
/// let backend_expert = factory.backend_selection();
/// let error_expert = factory.error_explanation();
/// let tactic_expert = factory.tactic_suggestion();
/// let compilation_expert = factory.compilation_guidance();
/// ```
pub struct ExpertFactory<'a> {
    store: &'a KnowledgeStore,
    embedder: &'a Embedder,
    tool_store: Option<&'a ToolKnowledgeStore>,
}

impl<'a> ExpertFactory<'a> {
    /// Create a new expert factory without ToolKnowledgeStore
    ///
    /// Experts created from this factory will use fallback/hardcoded values
    /// instead of the structured JSON knowledge base.
    pub fn new(store: &'a KnowledgeStore, embedder: &'a Embedder) -> Self {
        Self {
            store,
            embedder,
            tool_store: None,
        }
    }

    /// Create a new expert factory with ToolKnowledgeStore
    ///
    /// Experts created from this factory will use the structured JSON
    /// knowledge base for error patterns, tactics, comparisons, etc.
    pub fn with_tool_store(
        store: &'a KnowledgeStore,
        embedder: &'a Embedder,
        tool_store: &'a ToolKnowledgeStore,
    ) -> Self {
        Self {
            store,
            embedder,
            tool_store: Some(tool_store),
        }
    }

    /// Returns whether this factory has a ToolKnowledgeStore attached
    pub fn has_tool_store(&self) -> bool {
        self.tool_store.is_some()
    }

    /// Create a backend selection expert
    pub fn backend_selection(&self) -> BackendSelectionExpert<'a> {
        match self.tool_store {
            Some(ts) => BackendSelectionExpert::with_tool_store(self.store, self.embedder, ts),
            None => BackendSelectionExpert::new(self.store, self.embedder),
        }
    }

    /// Create an error explanation expert
    pub fn error_explanation(&self) -> ErrorExplanationExpert<'a> {
        match self.tool_store {
            Some(ts) => ErrorExplanationExpert::with_tool_store(self.store, self.embedder, ts),
            None => ErrorExplanationExpert::new(self.store, self.embedder),
        }
    }

    /// Create a tactic suggestion expert
    pub fn tactic_suggestion(&self) -> TacticSuggestionExpert<'a> {
        match self.tool_store {
            Some(ts) => TacticSuggestionExpert::with_tool_store(self.store, self.embedder, ts),
            None => TacticSuggestionExpert::new(self.store, self.embedder),
        }
    }

    /// Create a compilation guidance expert
    pub fn compilation_guidance(&self) -> CompilationGuidanceExpert<'a> {
        match self.tool_store {
            Some(ts) => CompilationGuidanceExpert::with_tool_store(self.store, self.embedder, ts),
            None => CompilationGuidanceExpert::new(self.store, self.embedder),
        }
    }

    /// Create a research recommendation expert
    ///
    /// This expert provides technique recommendations backed by ArXiv papers
    /// and academic research.
    pub fn research_recommendation(&self) -> ResearchRecommendationExpert<'a> {
        match self.tool_store {
            Some(ts) => {
                ResearchRecommendationExpert::with_tool_store(self.store, self.embedder, ts)
            }
            None => ResearchRecommendationExpert::new(self.store, self.embedder),
        }
    }

    /// Create all experts at once
    ///
    /// Returns a tuple of all five experts. Useful when you need
    /// to use multiple experts in the same context.
    pub fn all(
        &self,
    ) -> (
        BackendSelectionExpert<'a>,
        ErrorExplanationExpert<'a>,
        TacticSuggestionExpert<'a>,
        CompilationGuidanceExpert<'a>,
        ResearchRecommendationExpert<'a>,
    ) {
        (
            self.backend_selection(),
            self.error_explanation(),
            self.tactic_suggestion(),
            self.compilation_guidance(),
            self.research_recommendation(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EmbeddingModel;
    use std::path::PathBuf;

    fn create_test_store() -> KnowledgeStore {
        // Use default embedding dimensions for SentenceTransformers (384)
        KnowledgeStore::new(PathBuf::from("/tmp/test_store"), 384)
    }

    fn create_test_embedder() -> Embedder {
        // Use SentenceTransformers as the test model (local, no API key needed)
        Embedder::new(EmbeddingModel::SentenceTransformers)
    }

    #[test]
    fn test_factory_creation_without_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);

        assert!(!factory.has_tool_store());
    }

    #[test]
    fn test_factory_creation_with_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let tool_store = ToolKnowledgeStore::new();

        let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);

        assert!(factory.has_tool_store());
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: Verify factory state and constructor branches
    // ==========================================================================

    #[test]
    fn test_new_sets_tool_store_to_none() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);

        // Verify tool_store is actually None, not Some
        assert!(
            factory.tool_store.is_none(),
            "new() must set tool_store to None"
        );
    }

    #[test]
    fn test_with_tool_store_sets_tool_store_to_some() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let tool_store = ToolKnowledgeStore::new();

        let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);

        // Verify tool_store is actually Some, not None
        assert!(
            factory.tool_store.is_some(),
            "with_tool_store() must set tool_store to Some"
        );
    }

    #[test]
    fn test_has_tool_store_false_for_new() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);

        // Explicitly test false case
        assert!(
            !factory.has_tool_store(),
            "has_tool_store() must return false when created with new()"
        );
    }

    #[test]
    fn test_has_tool_store_true_for_with_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let tool_store = ToolKnowledgeStore::new();

        let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);

        // Explicitly test true case
        assert!(
            factory.has_tool_store(),
            "has_tool_store() must return true when created with with_tool_store()"
        );
    }

    #[test]
    fn test_factory_stores_correct_references() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);

        // Verify store and embedder are the same references
        assert!(std::ptr::eq(factory.store, &store));
        assert!(std::ptr::eq(factory.embedder, &embedder));
    }

    #[test]
    fn test_factory_with_tool_store_stores_correct_references() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let tool_store = ToolKnowledgeStore::new();

        let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);

        // Verify all references are correct
        assert!(std::ptr::eq(factory.store, &store));
        assert!(std::ptr::eq(factory.embedder, &embedder));
        assert!(std::ptr::eq(factory.tool_store.unwrap(), &tool_store));
    }

    #[test]
    fn test_create_backend_selection_expert() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);
        let _expert = factory.backend_selection();
        // Expert created successfully
    }

    #[test]
    fn test_create_error_explanation_expert() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);
        let _expert = factory.error_explanation();
        // Expert created successfully
    }

    #[test]
    fn test_create_tactic_suggestion_expert() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);
        let _expert = factory.tactic_suggestion();
        // Expert created successfully
    }

    #[test]
    fn test_create_compilation_guidance_expert() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);
        let _expert = factory.compilation_guidance();
        // Expert created successfully
    }

    #[test]
    fn test_create_all_experts() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);
        let (_backend, _error, _tactic, _compilation, _research) = factory.all();
        // All experts created successfully
    }

    #[test]
    fn test_create_experts_with_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let tool_store = ToolKnowledgeStore::new();

        let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);

        // Create all experts - they should use the tool store
        let _backend = factory.backend_selection();
        let _error = factory.error_explanation();
        let _tactic = factory.tactic_suggestion();
        let _compilation = factory.compilation_guidance();
        let _research = factory.research_recommendation();
        // All experts created with tool store
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: Verify factory method branches (Some vs None)
    // ==========================================================================

    #[test]
    fn test_backend_selection_none_branch_creates_expert_without_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        // Factory WITHOUT tool store
        let factory = ExpertFactory::new(&store, &embedder);
        assert!(factory.tool_store.is_none(), "Precondition: no tool store");

        // Should create expert using the None branch (BackendSelectionExpert::new)
        let expert = factory.backend_selection();

        // Verify expert was created (it would panic if wrong constructor called)
        // The expert should not have tool_store
        assert!(
            !expert.has_tool_store(),
            "Expert should not have tool store"
        );
    }

    #[test]
    fn test_backend_selection_some_branch_creates_expert_with_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let tool_store = ToolKnowledgeStore::new();

        // Factory WITH tool store
        let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);
        assert!(factory.tool_store.is_some(), "Precondition: has tool store");

        // Should create expert using the Some branch (BackendSelectionExpert::with_tool_store)
        let expert = factory.backend_selection();

        // Verify expert was created with tool store
        assert!(expert.has_tool_store(), "Expert should have tool store");
    }

    #[test]
    fn test_error_explanation_none_branch_creates_expert_without_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);
        assert!(factory.tool_store.is_none(), "Precondition: no tool store");

        let expert = factory.error_explanation();
        assert!(
            !expert.has_tool_store(),
            "Expert should not have tool store"
        );
    }

    #[test]
    fn test_error_explanation_some_branch_creates_expert_with_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let tool_store = ToolKnowledgeStore::new();

        let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);
        assert!(factory.tool_store.is_some(), "Precondition: has tool store");

        let expert = factory.error_explanation();
        assert!(expert.has_tool_store(), "Expert should have tool store");
    }

    #[test]
    fn test_tactic_suggestion_none_branch_creates_expert_without_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);
        assert!(factory.tool_store.is_none(), "Precondition: no tool store");

        let expert = factory.tactic_suggestion();
        assert!(
            !expert.has_tool_store(),
            "Expert should not have tool store"
        );
    }

    #[test]
    fn test_tactic_suggestion_some_branch_creates_expert_with_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let tool_store = ToolKnowledgeStore::new();

        let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);
        assert!(factory.tool_store.is_some(), "Precondition: has tool store");

        let expert = factory.tactic_suggestion();
        assert!(expert.has_tool_store(), "Expert should have tool store");
    }

    #[test]
    fn test_compilation_guidance_none_branch_creates_expert_without_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);
        assert!(factory.tool_store.is_none(), "Precondition: no tool store");

        let expert = factory.compilation_guidance();
        assert!(
            !expert.has_tool_store(),
            "Expert should not have tool store"
        );
    }

    #[test]
    fn test_compilation_guidance_some_branch_creates_expert_with_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let tool_store = ToolKnowledgeStore::new();

        let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);
        assert!(factory.tool_store.is_some(), "Precondition: has tool store");

        let expert = factory.compilation_guidance();
        assert!(expert.has_tool_store(), "Expert should have tool store");
    }

    #[test]
    fn test_research_recommendation_none_branch_creates_expert_without_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);
        assert!(factory.tool_store.is_none(), "Precondition: no tool store");

        let expert = factory.research_recommendation();
        assert!(
            !expert.has_tool_store(),
            "Expert should not have tool store"
        );
    }

    #[test]
    fn test_research_recommendation_some_branch_creates_expert_with_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let tool_store = ToolKnowledgeStore::new();

        let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);
        assert!(factory.tool_store.is_some(), "Precondition: has tool store");

        let expert = factory.research_recommendation();
        assert!(expert.has_tool_store(), "Expert should have tool store");
    }

    #[test]
    fn test_all_returns_five_experts_without_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);
        let (backend, error, tactic, compilation, research) = factory.all();

        // Verify all five are created without tool store
        assert!(!backend.has_tool_store());
        assert!(!error.has_tool_store());
        assert!(!tactic.has_tool_store());
        assert!(!compilation.has_tool_store());
        assert!(!research.has_tool_store());
    }

    #[test]
    fn test_all_returns_five_experts_with_tool_store() {
        let store = create_test_store();
        let embedder = create_test_embedder();
        let tool_store = ToolKnowledgeStore::new();

        let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);
        let (backend, error, tactic, compilation, research) = factory.all();

        // Verify all five are created with tool store
        assert!(backend.has_tool_store());
        assert!(error.has_tool_store());
        assert!(tactic.has_tool_store());
        assert!(compilation.has_tool_store());
        assert!(research.has_tool_store());
    }

    #[test]
    fn test_all_returns_correctly_ordered_tuple() {
        let store = create_test_store();
        let embedder = create_test_embedder();

        let factory = ExpertFactory::new(&store, &embedder);
        let (backend, error, tactic, compilation, _research) = factory.all();

        // The tuple order is: (BackendSelectionExpert, ErrorExplanationExpert, TacticSuggestionExpert, CompilationGuidanceExpert, ResearchRecommendationExpert)
        // Verify by checking each expert's characteristic behavior
        // Backend expert can get prefer_when
        let _ = backend.get_prefer_when(dashprove_backends::BackendId::Z3);
        // Error expert can analyze errors
        let _ = error.analyze_error("test", None);
        // Tactic expert can generate suggestions
        let _ = tactic.generate_suggestions("test", dashprove_backends::BackendId::Lean4);
        // Compilation expert can generate steps
        let _ = compilation.generate_steps("test", dashprove_backends::BackendId::Lean4);
        // Research expert - verified by has_tool_store above
    }

    // Integration tests using real JSON files
    mod integration {
        use super::*;
        use crate::expert::types::ExpertContext;
        use dashprove_backends::BackendId;
        use std::path::PathBuf;

        /// Get the path to the real JSON knowledge files
        fn get_resources_path() -> PathBuf {
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            PathBuf::from(manifest_dir).join("resources").join("tools")
        }

        #[tokio::test]
        async fn test_load_real_json_files() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge from resources");

            // Should have loaded many tools
            assert!(
                tool_store.len() > 100,
                "Expected > 100 tools, got {}",
                tool_store.len()
            );
        }

        #[tokio::test]
        async fn test_factory_with_real_json_z3() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            // Verify Z3 is loaded
            let z3 = tool_store.get("z3");
            assert!(z3.is_some(), "Z3 should be in the knowledge store");

            let z3 = z3.unwrap();
            assert_eq!(z3.name, "Z3");
            assert_eq!(z3.category, "smt_solver");
            assert!(!z3.tactics.is_empty(), "Z3 should have tactics");
            assert!(
                !z3.error_patterns.is_empty(),
                "Z3 should have error patterns"
            );
        }

        #[tokio::test]
        async fn test_error_explanation_with_real_json() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            let store = create_test_store();
            let embedder = create_test_embedder();

            let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);
            let expert = factory.error_explanation();

            // Test Z3 timeout error explanation
            let result = expert.explain("Error: timeout", Some(BackendId::Z3)).await;
            assert!(result.is_ok(), "Error explanation should succeed");

            let explanation = result.unwrap();
            assert!(!explanation.explanation.is_empty());
        }

        #[tokio::test]
        async fn test_tactic_suggestion_with_real_json() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            let store = create_test_store();
            let embedder = create_test_embedder();

            let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);
            let expert = factory.tactic_suggestion();

            // Test getting tactics for Z3
            let result = expert
                .suggest("prove arithmetic property", BackendId::Z3, None)
                .await;
            assert!(result.is_ok(), "Tactic suggestion should succeed");

            let suggestions = result.unwrap();
            // Should get some suggestions either from tool store or fallback
            assert!(!suggestions.is_empty(), "Should have tactic suggestions");
        }

        #[tokio::test]
        async fn test_backend_selection_with_real_json() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            let store = create_test_store();
            let embedder = create_test_embedder();

            let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);
            let expert = factory.backend_selection();

            let context = ExpertContext::default();
            let result = expert.recommend(&context).await;
            assert!(result.is_ok(), "Backend selection should succeed");

            let recommendation = result.unwrap();
            assert!(recommendation.confidence > 0.0);
        }

        #[tokio::test]
        async fn test_compilation_guidance_with_real_json() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            let store = create_test_store();
            let embedder = create_test_embedder();

            let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);
            let expert = factory.compilation_guidance();

            let spec = "theorem test: x + 0 = x";
            let result = expert.guide(spec, BackendId::Lean4).await;
            assert!(result.is_ok(), "Compilation guidance should succeed");

            let guidance = result.unwrap();
            assert!(!guidance.steps.is_empty(), "Should have compilation steps");
        }

        #[tokio::test]
        async fn test_all_experts_with_real_json() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            let store = create_test_store();
            let embedder = create_test_embedder();

            let factory = ExpertFactory::with_tool_store(&store, &embedder, &tool_store);

            // Create all experts at once
            let (backend_expert, error_expert, tactic_expert, compilation_expert, _research_expert) =
                factory.all();

            // Verify they can all perform their basic operations
            let _ = backend_expert.recommend(&ExpertContext::default()).await;
            let _ = error_expert.explain("error", Some(BackendId::Z3)).await;
            let _ = tactic_expert.suggest("goal", BackendId::Z3, None).await;
            let _ = compilation_expert.guide("spec", BackendId::Z3).await;
        }

        #[tokio::test]
        async fn test_tool_store_categories() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            // Check that various categories exist
            let smt_tools = tool_store.by_category("smt_solver");
            assert!(!smt_tools.is_empty(), "Should have SMT solver tools");

            let theorem_provers = tool_store.by_category("theorem_prover");
            // May or may not have theorem provers depending on JSON structure
            let _ = theorem_provers;
        }

        #[tokio::test]
        async fn test_tool_store_error_pattern_matching() {
            let resources_path = get_resources_path();
            if !resources_path.exists() {
                eprintln!("Skipping: resources/tools directory not found");
                return;
            }

            let tool_store = ToolKnowledgeStore::load_from_dir(&resources_path)
                .await
                .expect("Failed to load tool knowledge");

            // Test error pattern matching for Z3
            let matches = tool_store.find_error_fixes("z3", "Error: timeout exceeded");
            // Should find timeout pattern
            assert!(!matches.is_empty(), "Z3 should have timeout error pattern");
        }
    }
}
