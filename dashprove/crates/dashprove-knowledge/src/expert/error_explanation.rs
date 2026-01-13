//! Error explanation expert

use crate::embedding::Embedder;
use crate::store::{KnowledgeStore, VectorFilter};
use crate::tool_knowledge::ToolKnowledgeStore;
use crate::types::{ContentType, KnowledgeQuery};
use crate::Result;
use dashprove_backends::BackendId;

use super::types::{ErrorExplanation, Evidence, SuggestedFix};
use super::util::backend_id_to_tool_id;

/// Expert for error explanation
///
/// This expert uses both:
/// 1. The vector-based KnowledgeStore for semantic search over documentation
/// 2. The ToolKnowledgeStore for structured error pattern matching with regex support
pub struct ErrorExplanationExpert<'a> {
    store: &'a KnowledgeStore,
    embedder: &'a Embedder,
    tool_store: Option<&'a ToolKnowledgeStore>,
}

impl<'a> ErrorExplanationExpert<'a> {
    /// Create a new error explanation expert
    pub fn new(store: &'a KnowledgeStore, embedder: &'a Embedder) -> Self {
        Self {
            store,
            embedder,
            tool_store: None,
        }
    }

    /// Create a new error explanation expert with tool knowledge store
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

    /// Explain an error message
    pub async fn explain(
        &self,
        error_message: &str,
        backend: Option<BackendId>,
    ) -> Result<ErrorExplanation> {
        // Search for related error documentation
        let query_embedding = self.embedder.embed_text(error_message).await?;

        // Note: query is prepared for future use when we add full knowledge store search
        let _query = KnowledgeQuery {
            text: error_message.to_string(),
            backend,
            content_type: Some(ContentType::Errors),
            tags: vec![],
            limit: 10,
            include_papers: false,
            include_repos: false,
        };

        let filter = VectorFilter {
            backend,
            content_type: Some(ContentType::Errors),
            tags: vec![],
        };

        let chunks = self
            .store
            .vector_store
            .search(&query_embedding, 10, Some(filter));

        // Extract evidence
        let related_docs: Vec<Evidence> = chunks
            .iter()
            .take(5)
            .map(|c| Evidence {
                source: c.chunk.document_id.clone(),
                excerpt: c.chunk.content.chars().take(200).collect(),
                relevance: c.score,
            })
            .collect();

        // Generate explanation based on error patterns
        let (explanation, root_cause) = self.analyze_error(error_message, backend);
        let suggested_fixes = self.generate_fixes(error_message, backend);

        Ok(ErrorExplanation {
            original_error: error_message.to_string(),
            explanation,
            root_cause,
            suggested_fixes,
            related_docs,
            similar_issues: vec![],
        })
    }

    /// Analyze an error and return explanation and root cause
    ///
    /// First checks the ToolKnowledgeStore for regex-based pattern matching,
    /// then falls back to hardcoded patterns.
    pub fn analyze_error(&self, error: &str, backend: Option<BackendId>) -> (String, String) {
        // First try ToolKnowledgeStore if available and backend is known
        if let (Some(tool_store), Some(backend_id)) = (self.tool_store, backend) {
            let tool_id = backend_id_to_tool_id(backend_id);
            let matches = tool_store.find_error_fixes(&tool_id, error);

            if let Some(best_match) = matches.first() {
                // Found a match in the tool knowledge store
                let meaning = &best_match.pattern.meaning;
                let causes = if best_match.pattern.common_causes.is_empty() {
                    "Check the error context for more details.".to_string()
                } else {
                    best_match.pattern.common_causes.join("; ")
                };
                return (meaning.clone(), causes);
            }
        }

        // Fall back to hardcoded pattern matching
        let error_lower = error.to_lowercase();

        // Pattern matching for common errors
        if error_lower.contains("type") && error_lower.contains("mismatch") {
            return (
                "A type mismatch error indicates that the expected type differs from the actual type provided.".to_string(),
                "The expression produces a value of one type where a different type was expected.".to_string()
            );
        }

        if error_lower.contains("not found") || error_lower.contains("undefined") {
            return (
                "An undefined reference error indicates that a name or identifier cannot be resolved.".to_string(),
                "The referenced item may not be imported, may be misspelled, or may not exist in the current scope.".to_string()
            );
        }

        if error_lower.contains("timeout") || error_lower.contains("time limit") {
            return (
                "The verification timed out before completing.".to_string(),
                "The problem may be too large or complex for the current resource limits."
                    .to_string(),
            );
        }

        if error_lower.contains("counterexample") {
            return (
                "A counterexample was found, meaning the property does not hold.".to_string(),
                "The verification found a specific execution that violates the property."
                    .to_string(),
            );
        }

        // Backend-specific patterns (fallback for when tool store doesn't have the pattern)
        match backend {
            Some(BackendId::Lean4) => {
                if error_lower.contains("sorry") {
                    return (
                        "The proof contains 'sorry' which marks incomplete proofs.".to_string(),
                        "Complete the proof by replacing sorry with valid tactics.".to_string(),
                    );
                }
            }
            Some(BackendId::Kani) => {
                if error_lower.contains("unwinding") {
                    return (
                        "Kani hit its loop unwinding bound.".to_string(),
                        "The loop may be unbounded or the unwinding limit is too low.".to_string(),
                    );
                }
            }
            _ => {}
        }

        // Default explanation
        (
            format!("Error from {:?}: {}", backend, error),
            "Review the error message and consult backend documentation.".to_string(),
        )
    }

    /// Generate fix suggestions for an error
    ///
    /// First checks the ToolKnowledgeStore for structured fixes,
    /// then falls back to hardcoded suggestions.
    pub fn generate_fixes(&self, error: &str, backend: Option<BackendId>) -> Vec<SuggestedFix> {
        let mut fixes = Vec::new();

        // First try ToolKnowledgeStore if available and backend is known
        if let (Some(tool_store), Some(backend_id)) = (self.tool_store, backend) {
            let tool_id = backend_id_to_tool_id(backend_id);
            let matches = tool_store.find_error_fixes(&tool_id, error);

            for error_match in matches {
                for fix in &error_match.pattern.fixes {
                    fixes.push(SuggestedFix {
                        description: fix.clone(),
                        code_example: None, // Could be enhanced with examples from tactics
                        confidence: error_match.confidence,
                    });
                }
            }
        }

        // If we found fixes from tool store, return them
        if !fixes.is_empty() {
            return fixes;
        }

        // Fall back to hardcoded fixes
        let error_lower = error.to_lowercase();

        if error_lower.contains("type") && error_lower.contains("mismatch") {
            fixes.push(SuggestedFix {
                description:
                    "Check the types of all expressions and ensure they match the expected types."
                        .to_string(),
                code_example: None,
                confidence: 0.7,
            });
        }

        if error_lower.contains("timeout") {
            fixes.push(SuggestedFix {
                description: "Try simplifying the specification or increasing resource limits."
                    .to_string(),
                code_example: None,
                confidence: 0.6,
            });
            fixes.push(SuggestedFix {
                description: "Consider breaking the problem into smaller sub-problems.".to_string(),
                code_example: None,
                confidence: 0.5,
            });
        }

        if error_lower.contains("not found") {
            fixes.push(SuggestedFix {
                description: "Check that all required imports are present.".to_string(),
                code_example: None,
                confidence: 0.8,
            });
            fixes.push(SuggestedFix {
                description: "Verify spelling of identifiers.".to_string(),
                code_example: None,
                confidence: 0.7,
            });
        }

        // Backend-specific fixes
        match backend {
            Some(BackendId::Kani) => {
                if error_lower.contains("unwinding") {
                    fixes.push(SuggestedFix {
                        description: "Increase the unwind bound with #[kani::unwind(N)]"
                            .to_string(),
                        code_example: Some(
                            "#[kani::unwind(10)]\nfn check_loop() { ... }".to_string(),
                        ),
                        confidence: 0.75,
                    });
                }
            }
            Some(BackendId::Lean4) => {
                if error_lower.contains("sorry") {
                    fixes.push(SuggestedFix {
                        description: "Replace 'sorry' with a complete proof.".to_string(),
                        code_example: Some(
                            "-- Instead of: sorry\n-- Use: simp, rfl, exact, etc.".to_string(),
                        ),
                        confidence: 0.9,
                    });
                }
            }
            _ => {}
        }

        if fixes.is_empty() {
            fixes.push(SuggestedFix {
                description: "Consult the backend documentation for this error type.".to_string(),
                code_example: None,
                confidence: 0.4,
            });
        }

        fixes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_knowledge::{ErrorPattern, ToolKnowledge};
    use tempfile::TempDir;

    fn build_expert_components() -> (TempDir, KnowledgeStore, Embedder) {
        let temp_dir = TempDir::new().unwrap();
        let store = KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = Embedder::new(crate::EmbeddingModel::SentenceTransformers);
        (temp_dir, store, embedder)
    }

    fn assert_confidence(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < f32::EPSILON,
            "expected confidence {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_analyze_error_with_tool_store() {
        // Create a mock tool store with a Kani error pattern
        let mut tool_store = ToolKnowledgeStore::new();
        let kani_tool = ToolKnowledge {
            id: "kani".to_string(),
            name: "Kani".to_string(),
            category: "rust_formal_verification".to_string(),
            subcategory: None,
            description: "Model checker for Rust".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![],
            error_patterns: vec![ErrorPattern {
                pattern: "unwinding assertion loop .* iteration \\d+".to_string(),
                meaning: "Loop bound exceeded (from tool store)".to_string(),
                common_causes: vec!["Bound too small".to_string()],
                fixes: vec!["Increase #[kani::unwind(N)]".to_string()],
            }],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(kani_tool);

        // Create a minimal knowledge store
        let temp_dir = TempDir::new().unwrap();
        let store = KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        // Test with tool store
        let expert = ErrorExplanationExpert::with_tool_store(&store, &embedder, &tool_store);
        let (explanation, _cause) = expert.analyze_error(
            "Error: unwinding assertion loop 0 iteration 42 exceeded",
            Some(BackendId::Kani),
        );

        assert_eq!(explanation, "Loop bound exceeded (from tool store)");
    }

    #[test]
    fn test_generate_fixes_with_tool_store() {
        // Create a mock tool store with a Verus error pattern
        let mut tool_store = ToolKnowledgeStore::new();
        let verus_tool = ToolKnowledge {
            id: "verus".to_string(),
            name: "Verus".to_string(),
            category: "rust_formal_verification".to_string(),
            subcategory: None,
            description: "Deductive verifier for Rust".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![],
            error_patterns: vec![ErrorPattern {
                pattern: "postcondition not satisfied".to_string(),
                meaning: "Function doesn't guarantee its ensures clause".to_string(),
                common_causes: vec!["Missing case in implementation".to_string()],
                fixes: vec![
                    "Add missing case handling".to_string(),
                    "Weaken postcondition".to_string(),
                ],
            }],
            integration: None,
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(verus_tool);

        let temp_dir = TempDir::new().unwrap();
        let store = KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        let expert = ErrorExplanationExpert::with_tool_store(&store, &embedder, &tool_store);
        let fixes = expert.generate_fixes(
            "verification error: postcondition not satisfied at line 42",
            Some(BackendId::Verus),
        );

        assert_eq!(fixes.len(), 2);
        assert_eq!(fixes[0].description, "Add missing case handling");
        assert_eq!(fixes[1].description, "Weaken postcondition");
    }

    #[test]
    fn test_analyze_error_hardcoded_patterns_without_tool_store() {
        let (_temp_dir, store, embedder) = build_expert_components();
        let expert = ErrorExplanationExpert::new(&store, &embedder);

        let (explanation, root_cause) =
            expert.analyze_error("Type mismatch between u32 and u64", None);
        assert_eq!(
            explanation,
            "A type mismatch error indicates that the expected type differs from the actual type provided."
        );
        assert_eq!(
            root_cause,
            "The expression produces a value of one type where a different type was expected."
        );

        let (explanation, root_cause) =
            expert.analyze_error("symbol not found in current scope", None);
        assert_eq!(
            explanation,
            "An undefined reference error indicates that a name or identifier cannot be resolved."
        );
        assert_eq!(
            root_cause,
            "The referenced item may not be imported, may be misspelled, or may not exist in the current scope."
        );

        let (explanation, root_cause) = expert.analyze_error("solver timeout after 30s", None);
        assert_eq!(explanation, "The verification timed out before completing.");
        assert_eq!(
            root_cause,
            "The problem may be too large or complex for the current resource limits."
        );

        let (explanation, root_cause) =
            expert.analyze_error("counterexample found for property", None);
        assert_eq!(
            explanation,
            "A counterexample was found, meaning the property does not hold."
        );
        assert_eq!(
            root_cause,
            "The verification found a specific execution that violates the property."
        );
    }

    #[test]
    fn test_analyze_error_backend_specific_and_default_paths() {
        let (_temp_dir, store, embedder) = build_expert_components();
        let expert = ErrorExplanationExpert::new(&store, &embedder);

        let (lean_explanation, lean_root) = expert.analyze_error(
            "Proof still contains sorry placeholders",
            Some(BackendId::Lean4),
        );
        assert_eq!(
            lean_explanation,
            "The proof contains 'sorry' which marks incomplete proofs."
        );
        assert_eq!(
            lean_root,
            "Complete the proof by replacing sorry with valid tactics."
        );

        let (kani_explanation, kani_root) =
            expert.analyze_error("unwinding assertion loop 0 exceeded", Some(BackendId::Kani));
        assert_eq!(kani_explanation, "Kani hit its loop unwinding bound.");
        assert_eq!(
            kani_root,
            "The loop may be unbounded or the unwinding limit is too low."
        );

        let (fallback_explanation, fallback_root) =
            expert.analyze_error("ambiguous backend failure", Some(BackendId::TlaPlus));
        assert_eq!(
            fallback_explanation,
            "Error from Some(TlaPlus): ambiguous backend failure"
        );
        assert_eq!(
            fallback_root,
            "Review the error message and consult backend documentation."
        );
    }

    #[test]
    fn test_generate_fixes_hardcoded_suggestions_without_tool_store() {
        let (_temp_dir, store, embedder) = build_expert_components();
        let expert = ErrorExplanationExpert::new(&store, &embedder);

        let fixes = expert.generate_fixes("type mismatch for parameter", None);
        assert_eq!(fixes.len(), 1);
        assert_eq!(
            fixes[0].description,
            "Check the types of all expressions and ensure they match the expected types."
        );
        assert_confidence(fixes[0].confidence, 0.7);
        assert!(fixes[0].code_example.is_none());

        let timeout_fixes = expert.generate_fixes("verification timeout during solving", None);
        assert_eq!(timeout_fixes.len(), 2);
        assert_eq!(
            timeout_fixes[0].description,
            "Try simplifying the specification or increasing resource limits."
        );
        assert_confidence(timeout_fixes[0].confidence, 0.6);
        assert_eq!(
            timeout_fixes[1].description,
            "Consider breaking the problem into smaller sub-problems."
        );
        assert_confidence(timeout_fixes[1].confidence, 0.5);

        let undefined_fixes = expert.generate_fixes("module not found in path", None);
        assert_eq!(undefined_fixes.len(), 2);
        assert_eq!(
            undefined_fixes[0].description,
            "Check that all required imports are present."
        );
        assert_confidence(undefined_fixes[0].confidence, 0.8);
        assert_eq!(
            undefined_fixes[1].description,
            "Verify spelling of identifiers."
        );
        assert_confidence(undefined_fixes[1].confidence, 0.7);
    }

    #[test]
    fn test_generate_fixes_backend_specific_and_default() {
        let (_temp_dir, store, embedder) = build_expert_components();
        let expert = ErrorExplanationExpert::new(&store, &embedder);

        let kani_fixes = expert.generate_fixes(
            "unwinding assertion loop 0 iteration 42 exceeded",
            Some(BackendId::Kani),
        );
        assert_eq!(kani_fixes.len(), 1);
        assert_eq!(
            kani_fixes[0].description,
            "Increase the unwind bound with #[kani::unwind(N)]"
        );
        assert!(kani_fixes[0]
            .code_example
            .as_deref()
            .unwrap()
            .contains("#[kani::unwind(10)]"));
        assert_confidence(kani_fixes[0].confidence, 0.75);

        let lean_fixes = expert.generate_fixes("goal left as sorry", Some(BackendId::Lean4));
        assert_eq!(lean_fixes.len(), 1);
        assert_eq!(
            lean_fixes[0].description,
            "Replace 'sorry' with a complete proof."
        );
        assert!(lean_fixes[0]
            .code_example
            .as_deref()
            .unwrap()
            .contains("Instead of: sorry"));
        assert_confidence(lean_fixes[0].confidence, 0.9);

        let fallback_fixes =
            expert.generate_fixes("ambiguous backend failure", Some(BackendId::TlaPlus));
        assert_eq!(fallback_fixes.len(), 1);
        assert_eq!(
            fallback_fixes[0].description,
            "Consult the backend documentation for this error type."
        );
        assert_confidence(fallback_fixes[0].confidence, 0.4);
        assert!(fallback_fixes[0].code_example.is_none());
    }
}
