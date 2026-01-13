//! Tactic suggestion expert

use crate::embedding::Embedder;
use crate::store::KnowledgeStore;
use crate::tool_knowledge::ToolKnowledgeStore;
use crate::types::{ContentType, KnowledgeQuery};
use crate::Result;
use dashprove_backends::BackendId;

use super::types::TacticSuggestion;
use super::util::{backend_id_to_tool_id, backend_tactic_domain, extract_tactic_from_chunk};

/// Expert for tactic suggestions
///
/// This expert uses both:
/// 1. The vector-based KnowledgeStore for semantic search over documentation
/// 2. The ToolKnowledgeStore for structured tactic information from JSON files
pub struct TacticSuggestionExpert<'a> {
    store: &'a KnowledgeStore,
    embedder: &'a Embedder,
    tool_store: Option<&'a ToolKnowledgeStore>,
}

impl<'a> TacticSuggestionExpert<'a> {
    /// Create a new tactic suggestion expert
    pub fn new(store: &'a KnowledgeStore, embedder: &'a Embedder) -> Self {
        Self {
            store,
            embedder,
            tool_store: None,
        }
    }

    /// Create a new tactic suggestion expert with tool knowledge store
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

    /// Returns whether this expert has a ToolKnowledgeStore attached
    pub fn has_tool_store(&self) -> bool {
        self.tool_store.is_some()
    }

    /// Suggest tactics for a proof goal
    pub async fn suggest(
        &self,
        goal_description: &str,
        backend: BackendId,
        context: Option<&str>,
    ) -> Result<Vec<TacticSuggestion>> {
        // Build query
        let query_text = format!(
            "tactic for {} goal: {} context: {}",
            backend_tactic_domain(backend),
            goal_description,
            context.unwrap_or("")
        );

        let query_embedding = self.embedder.embed_text(&query_text).await?;

        let query = KnowledgeQuery {
            text: query_text.clone(),
            backend: Some(backend),
            content_type: Some(ContentType::Reference),
            tags: vec!["tactic".to_string()],
            limit: 10,
            include_papers: true,
            include_repos: false,
        };

        let results = self.store.search(&query, &query_embedding);

        // First try to get tactics from ToolKnowledgeStore if available
        let mut suggestions = self.get_tool_store_tactics(goal_description, backend);

        // If no tool store tactics, fall back to hardcoded suggestions
        if suggestions.is_empty() {
            suggestions = self.generate_suggestions(goal_description, backend);
        }

        // Enhance with search results
        for chunk in results.chunks.iter().take(3) {
            if chunk.score > 0.5 {
                suggestions.push(TacticSuggestion {
                    tactic: extract_tactic_from_chunk(&chunk.chunk.content),
                    backend,
                    when_to_use: "Based on similar proof patterns in knowledge base".to_string(),
                    expected_effect: "May help progress the proof".to_string(),
                    example: Some(chunk.chunk.content.chars().take(100).collect()),
                    confidence: chunk.score * 0.8,
                    alternatives: vec![],
                });
            }
        }

        // Sort by confidence
        suggestions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(suggestions)
    }

    /// Get tactic suggestions from the ToolKnowledgeStore
    fn get_tool_store_tactics(
        &self,
        goal_description: &str,
        backend: BackendId,
    ) -> Vec<TacticSuggestion> {
        let tool_store = match self.tool_store {
            Some(store) => store,
            None => return vec![],
        };

        let tool_id = backend_id_to_tool_id(backend);
        let tactics = tool_store.get_tactics(&tool_id);

        if tactics.is_empty() {
            return vec![];
        }

        let goal_lower = goal_description.to_lowercase();

        // Score tactics based on goal
        tactics
            .into_iter()
            .map(|tactic| {
                // Calculate relevance score based on when_to_use matching the goal
                let when_to_use = tactic.when_to_use.as_deref().unwrap_or("");
                let when_lower = when_to_use.to_lowercase();
                let desc_lower = tactic.description.to_lowercase();

                // Check for keyword matches
                let goal_terms: Vec<&str> = goal_lower.split_whitespace().collect();
                let matches: usize = goal_terms
                    .iter()
                    .filter(|term| {
                        when_lower.contains(*term)
                            || desc_lower.contains(*term)
                            || tactic.name.to_lowercase().contains(*term)
                    })
                    .count();

                // Also check for common patterns in the goal
                let pattern_match = if goal_lower.contains("=") || goal_lower.contains("equal") {
                    when_lower.contains("equal") || desc_lower.contains("reflexiv")
                } else if goal_lower.contains("forall") || goal_lower.contains("∀") {
                    when_lower.contains("forall")
                        || desc_lower.contains("introduc")
                        || desc_lower.contains("quantif")
                } else if goal_lower.contains("induct") {
                    desc_lower.contains("induct") || when_lower.contains("induct")
                } else {
                    false
                };

                // Calculate confidence based on matches
                let base_confidence = if matches > 0 || pattern_match {
                    0.6 + (matches as f32 * 0.1).min(0.3)
                } else {
                    0.4 // Low confidence fallback for general tactics
                };

                TacticSuggestion {
                    tactic: tactic.name.clone(),
                    backend,
                    when_to_use: when_to_use.to_string(),
                    expected_effect: tactic.description.clone(),
                    example: tactic.examples.first().cloned(),
                    confidence: base_confidence,
                    alternatives: vec![],
                }
            })
            .collect()
    }

    /// Generate tactic suggestions based on goal and backend
    pub fn generate_suggestions(
        &self,
        goal_description: &str,
        backend: BackendId,
    ) -> Vec<TacticSuggestion> {
        let goal_lower = goal_description.to_lowercase();
        let mut suggestions = Vec::new();

        match backend {
            BackendId::Lean4 => {
                if goal_lower.contains("=") || goal_lower.contains("equal") {
                    suggestions.push(TacticSuggestion {
                        tactic: "rfl".to_string(),
                        backend,
                        when_to_use: "When both sides of equality are definitionally equal"
                            .to_string(),
                        expected_effect: "Closes goals where LHS and RHS reduce to the same term"
                            .to_string(),
                        example: Some("example : 2 + 2 = 4 := by rfl".to_string()),
                        confidence: 0.8,
                        alternatives: vec!["simp".to_string(), "ring".to_string()],
                    });
                    suggestions.push(TacticSuggestion {
                        tactic: "simp".to_string(),
                        backend,
                        when_to_use: "When simplification lemmas can reduce the goal".to_string(),
                        expected_effect: "Applies rewrite rules to simplify expressions"
                            .to_string(),
                        example: Some("example : x + 0 = x := by simp".to_string()),
                        confidence: 0.75,
                        alternatives: vec!["simp_all".to_string(), "simp only [...]".to_string()],
                    });
                }
                if goal_lower.contains("forall")
                    || goal_lower.contains("∀")
                    || goal_lower.contains("->")
                {
                    suggestions.push(TacticSuggestion {
                        tactic: "intro".to_string(),
                        backend,
                        when_to_use: "When goal starts with ∀ or →".to_string(),
                        expected_effect: "Introduces hypotheses into context".to_string(),
                        example: Some("example : ∀ n, n = n := by intro n; rfl".to_string()),
                        confidence: 0.85,
                        alternatives: vec!["intros".to_string()],
                    });
                }
                if goal_lower.contains("induct") || goal_lower.contains("nat") {
                    suggestions.push(TacticSuggestion {
                        tactic: "induction".to_string(),
                        backend,
                        when_to_use: "When proving properties about recursive structures"
                            .to_string(),
                        expected_effect: "Creates base case and inductive case goals".to_string(),
                        example: Some(
                            "induction n with | zero => ... | succ n ih => ...".to_string(),
                        ),
                        confidence: 0.7,
                        alternatives: vec!["cases".to_string(), "rcases".to_string()],
                    });
                }
                // Fallback for Lean4 if no specific tactic matched
                if suggestions.is_empty() {
                    suggestions.push(TacticSuggestion {
                        tactic: "simp".to_string(),
                        backend,
                        when_to_use: "As a first attempt for general simplification".to_string(),
                        expected_effect: "Applies simplification lemmas".to_string(),
                        example: Some("by simp".to_string()),
                        confidence: 0.5,
                        alternatives: vec!["decide".to_string(), "trivial".to_string()],
                    });
                }
            }
            BackendId::Coq => {
                if goal_lower.contains("=") {
                    suggestions.push(TacticSuggestion {
                        tactic: "reflexivity".to_string(),
                        backend,
                        when_to_use: "When both sides are convertible".to_string(),
                        expected_effect: "Proves x = x goals".to_string(),
                        example: Some("Proof. reflexivity. Qed.".to_string()),
                        confidence: 0.8,
                        alternatives: vec!["auto".to_string(), "simpl".to_string()],
                    });
                }
                if goal_lower.contains("forall") {
                    suggestions.push(TacticSuggestion {
                        tactic: "intros".to_string(),
                        backend,
                        when_to_use: "When goal has forall or implication".to_string(),
                        expected_effect: "Moves quantified variables to context".to_string(),
                        example: Some("intros x y H.".to_string()),
                        confidence: 0.85,
                        alternatives: vec!["intro".to_string()],
                    });
                }
                // Fallback for Coq if no specific tactic matched
                if suggestions.is_empty() {
                    suggestions.push(TacticSuggestion {
                        tactic: "auto".to_string(),
                        backend,
                        when_to_use: "As a first attempt for automatic reasoning".to_string(),
                        expected_effect: "Applies hints database automatically".to_string(),
                        example: Some("Proof. auto. Qed.".to_string()),
                        confidence: 0.5,
                        alternatives: vec!["trivial".to_string(), "easy".to_string()],
                    });
                }
            }
            BackendId::Isabelle => {
                suggestions.push(TacticSuggestion {
                    tactic: "auto".to_string(),
                    backend,
                    when_to_use: "As a first attempt on most goals".to_string(),
                    expected_effect: "Applies simplification and classical reasoning".to_string(),
                    example: Some("by auto".to_string()),
                    confidence: 0.7,
                    alternatives: vec!["simp".to_string(), "blast".to_string()],
                });
            }
            BackendId::TlaPlus => {
                suggestions.push(TacticSuggestion {
                    tactic: "BY DEF".to_string(),
                    backend,
                    when_to_use: "When need to expand definitions".to_string(),
                    expected_effect: "Expands operator definitions".to_string(),
                    example: Some("<1>1. Init => Inv BY DEF Init, Inv".to_string()),
                    confidence: 0.75,
                    alternatives: vec!["OBVIOUS".to_string()],
                });
            }
            _ => {
                // Generic suggestion for other backends
                suggestions.push(TacticSuggestion {
                    tactic: "auto".to_string(),
                    backend,
                    when_to_use: "Try automatic proof search".to_string(),
                    expected_effect: "Attempts to solve goal automatically".to_string(),
                    example: None,
                    confidence: 0.5,
                    alternatives: vec![],
                });
            }
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_knowledge::{Tactic, ToolKnowledge, ToolKnowledgeStore};

    /// Helper to create a minimal test environment (store, embedder)
    fn create_test_env() -> (
        tempfile::TempDir,
        crate::store::KnowledgeStore,
        crate::embedding::Embedder,
    ) {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);
        (temp_dir, store, embedder)
    }

    /// Helper to create a tool store with Lean4 tactics
    fn create_lean4_tool_store() -> ToolKnowledgeStore {
        let mut tool_store = ToolKnowledgeStore::new();
        let lean4_tool = ToolKnowledge {
            id: "lean4".to_string(),
            name: "Lean 4".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "Interactive theorem prover".to_string(),
            long_description: None,
            capabilities: vec!["tactic_proofs".to_string()],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![
                Tactic {
                    name: "simp".to_string(),
                    description: "Simplification tactic for equalities".to_string(),
                    syntax: Some("simp [...]".to_string()),
                    when_to_use: Some("Use for equality goals".to_string()),
                    examples: vec!["simp [add_comm]".to_string()],
                },
                Tactic {
                    name: "induction".to_string(),
                    description: "Proof by induction on recursive types".to_string(),
                    syntax: Some("induction x with ...".to_string()),
                    when_to_use: Some("Use for induction proofs".to_string()),
                    examples: vec!["induction n with | zero => ... | succ n ih => ...".to_string()],
                },
            ],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(lean4_tool);
        tool_store
    }

    #[test]
    fn test_get_tool_store_tactics_with_matching_goal() {
        let tool_store = create_lean4_tool_store();
        let (_temp_dir, store, embedder) = create_test_env();

        let expert = TacticSuggestionExpert::with_tool_store(&store, &embedder, &tool_store);

        // Test equality goal matches simp tactic
        let suggestions = expert.get_tool_store_tactics("prove x = y", BackendId::Lean4);
        assert!(!suggestions.is_empty());
        let has_simp = suggestions.iter().any(|s| s.tactic == "simp");
        assert!(has_simp, "Should find simp tactic for equality goal");

        // Test induction goal
        let suggestions = expert.get_tool_store_tactics("prove by induction", BackendId::Lean4);
        assert!(!suggestions.is_empty());
        let has_induction = suggestions.iter().any(|s| s.tactic == "induction");
        assert!(
            has_induction,
            "Should find induction tactic for induction goal"
        );
    }

    #[test]
    fn test_get_tool_store_tactics_empty_when_no_store() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.get_tool_store_tactics("prove x = y", BackendId::Lean4);
        assert!(
            suggestions.is_empty(),
            "Should return empty when no tool store"
        );
    }

    #[test]
    fn test_generate_suggestions_fallback() {
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        // Test Lean4 equality goal
        let suggestions = expert.generate_suggestions("prove 2 + 2 = 4", BackendId::Lean4);
        assert!(!suggestions.is_empty());
        let has_equality_tactic = suggestions
            .iter()
            .any(|s| s.tactic == "rfl" || s.tactic == "simp");
        assert!(
            has_equality_tactic,
            "Should suggest rfl or simp for equality goal"
        );

        // Test Coq forall goal
        let suggestions = expert.generate_suggestions("forall x, P x", BackendId::Coq);
        assert!(!suggestions.is_empty());
        let has_intros = suggestions.iter().any(|s| s.tactic == "intros");
        assert!(has_intros, "Should suggest intros for forall goal in Coq");

        // Test TlaPlus
        let suggestions = expert.generate_suggestions("prove invariant", BackendId::TlaPlus);
        assert!(!suggestions.is_empty());
        let has_by_def = suggestions.iter().any(|s| s.tactic == "BY DEF");
        assert!(has_by_def, "Should suggest BY DEF for TLA+");
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: Confidence score calculations
    // ==========================================================================

    #[test]
    fn test_confidence_score_with_keyword_matches() {
        // This test catches mutations to the confidence formula:
        // base_confidence = 0.6 + (matches as f32 * 0.1).min(0.3)
        let mut tool_store = ToolKnowledgeStore::new();
        let lean4_tool = ToolKnowledge {
            id: "lean4".to_string(),
            name: "Lean 4".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "Interactive theorem prover".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![Tactic {
                name: "test_tactic".to_string(),
                description: "prove equality reflexive".to_string(),
                syntax: None,
                when_to_use: Some("Use when proving".to_string()),
                examples: vec![],
            }],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(lean4_tool);

        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::with_tool_store(&store, &embedder, &tool_store);

        // Goal with multiple keyword matches ("prove" and "equality")
        // matches = 2, so confidence = 0.6 + 0.2 = 0.8
        let suggestions = expert.get_tool_store_tactics("prove equality", BackendId::Lean4);
        assert!(!suggestions.is_empty());
        let confidence = suggestions[0].confidence;
        // Must be > 0.6 (base) and <= 0.9 (max)
        assert!(
            confidence > 0.6,
            "Confidence with 2 matches should be > 0.6, got {}",
            confidence
        );
        assert!(
            confidence <= 0.9,
            "Confidence with 2 matches should be <= 0.9, got {}",
            confidence
        );
    }

    #[test]
    fn test_confidence_score_no_matches_fallback() {
        // This test catches mutations to the fallback confidence (0.4)
        let mut tool_store = ToolKnowledgeStore::new();
        let lean4_tool = ToolKnowledge {
            id: "lean4".to_string(),
            name: "Lean 4".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "Interactive theorem prover".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![Tactic {
                name: "zzz_tactic".to_string(),
                // Use terms that won't match any goal keywords we use
                description: "handles alpha beta gamma".to_string(),
                syntax: None,
                when_to_use: Some("delta epsilon zeta".to_string()),
                examples: vec![],
            }],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(lean4_tool);

        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::with_tool_store(&store, &embedder, &tool_store);

        // Goal with no keyword matches - use unique terms
        let suggestions = expert.get_tool_store_tactics("xyz qrs tuv", BackendId::Lean4);
        assert!(!suggestions.is_empty());
        let confidence = suggestions[0].confidence;
        // Must be exactly 0.4 (fallback)
        assert!(
            (confidence - 0.4).abs() < f32::EPSILON,
            "Confidence with no matches should be 0.4, got {}",
            confidence
        );
    }

    #[test]
    fn test_confidence_cap_at_three_matches() {
        // This test catches mutations to the .min(0.3) cap
        // With 4+ matches, confidence should still cap at 0.6 + 0.3 = 0.9
        let mut tool_store = ToolKnowledgeStore::new();
        let lean4_tool = ToolKnowledge {
            id: "lean4".to_string(),
            name: "Lean 4".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "Interactive theorem prover".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![Tactic {
                name: "multi_match".to_string(),
                description: "prove equal forall induction nat".to_string(),
                syntax: None,
                when_to_use: Some("prove equal forall induction nat".to_string()),
                examples: vec![],
            }],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(lean4_tool);

        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::with_tool_store(&store, &embedder, &tool_store);

        // Goal with 5 keyword matches - should cap at 0.9
        let suggestions =
            expert.get_tool_store_tactics("prove equal forall induction nat", BackendId::Lean4);
        assert!(!suggestions.is_empty());
        let confidence = suggestions[0].confidence;
        // Should be capped at 0.9
        assert!(
            (confidence - 0.9).abs() < f32::EPSILON,
            "Confidence should cap at 0.9, got {}",
            confidence
        );
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: Pattern matching in get_tool_store_tactics
    // ==========================================================================

    #[test]
    fn test_pattern_match_equality_symbol() {
        // This test catches mutations to: goal_lower.contains("=") || goal_lower.contains("equal")
        let mut tool_store = ToolKnowledgeStore::new();
        let lean4_tool = ToolKnowledge {
            id: "lean4".to_string(),
            name: "Lean 4".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "theorem prover".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![Tactic {
                name: "rfl".to_string(),
                description: "reflexivity".to_string(),
                syntax: None,
                when_to_use: Some("equal reflexive".to_string()),
                examples: vec![],
            }],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(lean4_tool);

        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::with_tool_store(&store, &embedder, &tool_store);

        // Test with "=" symbol - pattern_match should be true
        let suggestions = expert.get_tool_store_tactics("x = y", BackendId::Lean4);
        assert!(!suggestions.is_empty());
        // Confidence should be elevated due to pattern_match
        let confidence = suggestions[0].confidence;
        assert!(
            confidence >= 0.6,
            "Equality pattern with '=' should have confidence >= 0.6, got {}",
            confidence
        );
    }

    #[test]
    fn test_pattern_match_forall_unicode() {
        // This test catches mutations to: goal_lower.contains("forall") || goal_lower.contains("∀")
        let mut tool_store = ToolKnowledgeStore::new();
        let lean4_tool = ToolKnowledge {
            id: "lean4".to_string(),
            name: "Lean 4".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "theorem prover".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![Tactic {
                name: "intro".to_string(),
                description: "introduce hypothesis".to_string(),
                syntax: None,
                when_to_use: Some("forall quantifier".to_string()),
                examples: vec![],
            }],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(lean4_tool);

        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::with_tool_store(&store, &embedder, &tool_store);

        // Test with unicode "∀" - pattern_match should be true
        let suggestions = expert.get_tool_store_tactics("∀ x, P x", BackendId::Lean4);
        assert!(!suggestions.is_empty());
        let confidence = suggestions[0].confidence;
        assert!(
            confidence >= 0.6,
            "Forall pattern with '∀' should have confidence >= 0.6, got {}",
            confidence
        );
    }

    #[test]
    fn test_pattern_match_induction_keyword() {
        // This test catches mutations to: goal_lower.contains("induct")
        let mut tool_store = ToolKnowledgeStore::new();
        let lean4_tool = ToolKnowledge {
            id: "lean4".to_string(),
            name: "Lean 4".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "theorem prover".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![Tactic {
                name: "induction".to_string(),
                description: "proof by induction".to_string(),
                syntax: None,
                when_to_use: Some("induction hypothesis".to_string()),
                examples: vec![],
            }],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(lean4_tool);

        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::with_tool_store(&store, &embedder, &tool_store);

        // Test with "induction" keyword
        let suggestions =
            expert.get_tool_store_tactics("prove by induction on n", BackendId::Lean4);
        assert!(!suggestions.is_empty());
        let confidence = suggestions[0].confidence;
        assert!(
            confidence >= 0.6,
            "Induction pattern should have confidence >= 0.6, got {}",
            confidence
        );
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: generate_suggestions backend branches
    // ==========================================================================

    #[test]
    fn test_generate_suggestions_lean4_equality_produces_rfl() {
        // This test catches mutations to Lean4 equality branch
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("x = x", BackendId::Lean4);
        let rfl = suggestions.iter().find(|s| s.tactic == "rfl");
        assert!(rfl.is_some(), "Lean4 equality goal should suggest rfl");

        let rfl = rfl.unwrap();
        assert!(
            (rfl.confidence - 0.8).abs() < f32::EPSILON,
            "rfl confidence should be 0.8, got {}",
            rfl.confidence
        );
        assert!(
            rfl.when_to_use.contains("definitionally"),
            "rfl when_to_use should mention 'definitionally'"
        );
    }

    #[test]
    fn test_generate_suggestions_lean4_forall_produces_intro() {
        // This test catches mutations to Lean4 forall branch
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("∀ n, n = n", BackendId::Lean4);
        let intro = suggestions.iter().find(|s| s.tactic == "intro");
        assert!(intro.is_some(), "Lean4 forall goal should suggest intro");

        let intro = intro.unwrap();
        assert!(
            (intro.confidence - 0.85).abs() < f32::EPSILON,
            "intro confidence should be 0.85, got {}",
            intro.confidence
        );
    }

    #[test]
    fn test_generate_suggestions_lean4_arrow_produces_intro() {
        // This test catches mutations to: goal_lower.contains("->")
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("P -> Q", BackendId::Lean4);
        let intro = suggestions.iter().find(|s| s.tactic == "intro");
        assert!(
            intro.is_some(),
            "Lean4 implication goal should suggest intro"
        );
    }

    #[test]
    fn test_generate_suggestions_lean4_nat_produces_induction() {
        // This test catches mutations to: goal_lower.contains("nat")
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("prove property for Nat", BackendId::Lean4);
        let induction = suggestions.iter().find(|s| s.tactic == "induction");
        assert!(
            induction.is_some(),
            "Lean4 Nat goal should suggest induction"
        );

        let induction = induction.unwrap();
        assert!(
            (induction.confidence - 0.7).abs() < f32::EPSILON,
            "induction confidence should be 0.7, got {}",
            induction.confidence
        );
    }

    #[test]
    fn test_generate_suggestions_lean4_fallback_to_simp() {
        // This test catches mutations to Lean4 fallback branch
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        // Goal with no specific patterns
        let suggestions = expert.generate_suggestions("something else entirely", BackendId::Lean4);
        assert!(!suggestions.is_empty());
        let simp = suggestions.iter().find(|s| s.tactic == "simp");
        assert!(simp.is_some(), "Lean4 fallback should suggest simp");

        let simp = simp.unwrap();
        assert!(
            (simp.confidence - 0.5).abs() < f32::EPSILON,
            "simp fallback confidence should be 0.5, got {}",
            simp.confidence
        );
    }

    #[test]
    fn test_generate_suggestions_coq_equality_produces_reflexivity() {
        // This test catches mutations to Coq equality branch
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("x = x", BackendId::Coq);
        let reflexivity = suggestions.iter().find(|s| s.tactic == "reflexivity");
        assert!(
            reflexivity.is_some(),
            "Coq equality goal should suggest reflexivity"
        );

        let reflexivity = reflexivity.unwrap();
        assert!(
            (reflexivity.confidence - 0.8).abs() < f32::EPSILON,
            "reflexivity confidence should be 0.8, got {}",
            reflexivity.confidence
        );
    }

    #[test]
    fn test_generate_suggestions_coq_forall_produces_intros() {
        // This test catches mutations to Coq forall branch
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("forall x, P x", BackendId::Coq);
        let intros = suggestions.iter().find(|s| s.tactic == "intros");
        assert!(intros.is_some(), "Coq forall goal should suggest intros");

        let intros = intros.unwrap();
        assert!(
            (intros.confidence - 0.85).abs() < f32::EPSILON,
            "intros confidence should be 0.85, got {}",
            intros.confidence
        );
    }

    #[test]
    fn test_generate_suggestions_coq_fallback_to_auto() {
        // This test catches mutations to Coq fallback branch
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        // Goal with no specific patterns
        let suggestions = expert.generate_suggestions("something else entirely", BackendId::Coq);
        assert!(!suggestions.is_empty());
        let auto = suggestions.iter().find(|s| s.tactic == "auto");
        assert!(auto.is_some(), "Coq fallback should suggest auto");

        let auto = auto.unwrap();
        assert!(
            (auto.confidence - 0.5).abs() < f32::EPSILON,
            "auto fallback confidence should be 0.5, got {}",
            auto.confidence
        );
    }

    #[test]
    fn test_generate_suggestions_isabelle_produces_auto() {
        // This test catches mutations to Isabelle branch
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("any goal", BackendId::Isabelle);
        assert!(!suggestions.is_empty());
        let auto = suggestions.iter().find(|s| s.tactic == "auto");
        assert!(auto.is_some(), "Isabelle should suggest auto");

        let auto = auto.unwrap();
        assert!(
            (auto.confidence - 0.7).abs() < f32::EPSILON,
            "Isabelle auto confidence should be 0.7, got {}",
            auto.confidence
        );
    }

    #[test]
    fn test_generate_suggestions_tlaplus_produces_by_def() {
        // This test catches mutations to TLA+ branch
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("prove invariant", BackendId::TlaPlus);
        assert!(!suggestions.is_empty());
        let by_def = suggestions.iter().find(|s| s.tactic == "BY DEF");
        assert!(by_def.is_some(), "TLA+ should suggest BY DEF");

        let by_def = by_def.unwrap();
        assert!(
            (by_def.confidence - 0.75).abs() < f32::EPSILON,
            "TLA+ BY DEF confidence should be 0.75, got {}",
            by_def.confidence
        );
    }

    #[test]
    fn test_generate_suggestions_default_backend_produces_auto() {
        // This test catches mutations to the default (_) branch
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        // Use a backend that falls into the default case
        let suggestions = expert.generate_suggestions("any goal", BackendId::Kani);
        assert!(!suggestions.is_empty());
        let auto = suggestions.iter().find(|s| s.tactic == "auto");
        assert!(auto.is_some(), "Default backend should suggest auto");

        let auto = auto.unwrap();
        assert!(
            (auto.confidence - 0.5).abs() < f32::EPSILON,
            "Default auto confidence should be 0.5, got {}",
            auto.confidence
        );
        assert!(
            auto.example.is_none(),
            "Default auto should have no example"
        );
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: Specific field values
    // ==========================================================================

    #[test]
    fn test_lean4_simp_alternatives() {
        // This test catches mutations to alternatives vec
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("x = y", BackendId::Lean4);
        let simp = suggestions.iter().find(|s| s.tactic == "simp");
        assert!(simp.is_some());

        let simp = simp.unwrap();
        assert!(
            simp.alternatives.contains(&"simp_all".to_string()),
            "simp alternatives should include simp_all"
        );
        assert!(
            simp.alternatives.contains(&"simp only [...]".to_string()),
            "simp alternatives should include 'simp only [...]'"
        );
    }

    #[test]
    fn test_lean4_intro_alternatives() {
        // This test catches mutations to intro alternatives
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("∀ n, P n", BackendId::Lean4);
        let intro = suggestions.iter().find(|s| s.tactic == "intro");
        assert!(intro.is_some());

        let intro = intro.unwrap();
        assert!(
            intro.alternatives.contains(&"intros".to_string()),
            "intro alternatives should include intros"
        );
    }

    #[test]
    fn test_lean4_induction_alternatives() {
        // This test catches mutations to induction alternatives
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("prove by induction", BackendId::Lean4);
        let induction = suggestions.iter().find(|s| s.tactic == "induction");
        assert!(induction.is_some());

        let induction = induction.unwrap();
        assert!(
            induction.alternatives.contains(&"cases".to_string()),
            "induction alternatives should include cases"
        );
        assert!(
            induction.alternatives.contains(&"rcases".to_string()),
            "induction alternatives should include rcases"
        );
    }

    #[test]
    fn test_coq_reflexivity_alternatives() {
        // This test catches mutations to Coq reflexivity alternatives
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("x = x", BackendId::Coq);
        let reflexivity = suggestions.iter().find(|s| s.tactic == "reflexivity");
        assert!(reflexivity.is_some());

        let reflexivity = reflexivity.unwrap();
        assert!(
            reflexivity.alternatives.contains(&"auto".to_string()),
            "reflexivity alternatives should include auto"
        );
        assert!(
            reflexivity.alternatives.contains(&"simpl".to_string()),
            "reflexivity alternatives should include simpl"
        );
    }

    #[test]
    fn test_isabelle_auto_alternatives() {
        // This test catches mutations to Isabelle auto alternatives
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("any goal", BackendId::Isabelle);
        let auto = suggestions.iter().find(|s| s.tactic == "auto");
        assert!(auto.is_some());

        let auto = auto.unwrap();
        assert!(
            auto.alternatives.contains(&"simp".to_string()),
            "Isabelle auto alternatives should include simp"
        );
        assert!(
            auto.alternatives.contains(&"blast".to_string()),
            "Isabelle auto alternatives should include blast"
        );
    }

    #[test]
    fn test_tlaplus_by_def_alternatives() {
        // This test catches mutations to TLA+ BY DEF alternatives
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("prove inv", BackendId::TlaPlus);
        let by_def = suggestions.iter().find(|s| s.tactic == "BY DEF");
        assert!(by_def.is_some());

        let by_def = by_def.unwrap();
        assert!(
            by_def.alternatives.contains(&"OBVIOUS".to_string()),
            "BY DEF alternatives should include OBVIOUS"
        );
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: TacticSuggestion field content
    // ==========================================================================

    #[test]
    fn test_tactic_suggestion_backend_field_matches_input() {
        // This test catches mutations where backend field is set incorrectly
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        for backend in [
            BackendId::Lean4,
            BackendId::Coq,
            BackendId::Isabelle,
            BackendId::TlaPlus,
            BackendId::Kani,
        ] {
            let suggestions = expert.generate_suggestions("any goal", backend);
            for suggestion in &suggestions {
                assert_eq!(
                    suggestion.backend, backend,
                    "All suggestions should have backend = {:?}",
                    backend
                );
            }
        }
    }

    #[test]
    fn test_tactic_suggestion_example_contains_tactic_name() {
        // This test catches mutations where example doesn't match tactic
        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::new(&store, &embedder);

        let suggestions = expert.generate_suggestions("x = x", BackendId::Lean4);
        let rfl = suggestions.iter().find(|s| s.tactic == "rfl");
        assert!(rfl.is_some());

        let rfl = rfl.unwrap();
        assert!(
            rfl.example.as_ref().unwrap().contains("rfl"),
            "rfl example should contain 'rfl'"
        );
    }

    // ==========================================================================
    // MUTATION-KILLING TESTS: Empty tactics return
    // ==========================================================================

    #[test]
    fn test_get_tool_store_tactics_empty_when_tool_has_no_tactics() {
        let mut tool_store = ToolKnowledgeStore::new();
        let lean4_tool = ToolKnowledge {
            id: "lean4".to_string(),
            name: "Lean 4".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "Interactive theorem prover".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![], // No tactics
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(lean4_tool);

        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::with_tool_store(&store, &embedder, &tool_store);

        let suggestions = expert.get_tool_store_tactics("prove x = y", BackendId::Lean4);
        assert!(
            suggestions.is_empty(),
            "Should return empty when tool has no tactics"
        );
    }

    #[test]
    fn test_get_tool_store_tactics_with_none_when_to_use() {
        // This test catches mutations to: when_to_use.as_deref().unwrap_or("")
        let mut tool_store = ToolKnowledgeStore::new();
        let lean4_tool = ToolKnowledge {
            id: "lean4".to_string(),
            name: "Lean 4".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "Interactive theorem prover".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![Tactic {
                name: "test".to_string(),
                description: "test tactic".to_string(),
                syntax: None,
                when_to_use: None, // None value
                examples: vec![],
            }],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(lean4_tool);

        let (_temp_dir, store, embedder) = create_test_env();
        let expert = TacticSuggestionExpert::with_tool_store(&store, &embedder, &tool_store);

        // Should not panic when when_to_use is None
        let suggestions = expert.get_tool_store_tactics("prove goal", BackendId::Lean4);
        assert!(!suggestions.is_empty());
        // when_to_use should be empty string
        assert_eq!(suggestions[0].when_to_use, "");
    }
}
