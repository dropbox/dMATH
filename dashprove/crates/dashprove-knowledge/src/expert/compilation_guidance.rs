//! Compilation guidance expert

use crate::embedding::Embedder;
use crate::store::KnowledgeStore;
use crate::tool_knowledge::ToolKnowledgeStore;
use crate::types::{ContentType, KnowledgeQuery};
use crate::Result;
use dashprove_backends::BackendId;

use super::types::{CompilationGuidance, CompilationStep, Evidence};
use super::util::backend_id_to_tool_id;

/// Expert for compilation guidance
///
/// This expert uses both:
/// 1. The vector-based KnowledgeStore for semantic search over documentation
/// 2. The ToolKnowledgeStore for structured compilation info (installation, tactics, docs)
pub struct CompilationGuidanceExpert<'a> {
    store: &'a KnowledgeStore,
    embedder: &'a Embedder,
    tool_store: Option<&'a ToolKnowledgeStore>,
}

impl<'a> CompilationGuidanceExpert<'a> {
    /// Create a new compilation guidance expert
    pub fn new(store: &'a KnowledgeStore, embedder: &'a Embedder) -> Self {
        Self {
            store,
            embedder,
            tool_store: None,
        }
    }

    /// Create a new compilation guidance expert with tool knowledge store
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

    /// Get guidance for compiling to a backend
    pub async fn guide(
        &self,
        specification: &str,
        target_backend: BackendId,
    ) -> Result<CompilationGuidance> {
        // Search for relevant compilation documentation
        let query_text = format!(
            "compile to {:?} specification: {}",
            target_backend,
            specification.chars().take(200).collect::<String>()
        );

        let query_embedding = self.embedder.embed_text(&query_text).await?;

        let query = KnowledgeQuery {
            text: query_text.clone(),
            backend: Some(target_backend),
            content_type: Some(ContentType::Tutorial),
            tags: vec!["compilation".to_string()],
            limit: 10,
            include_papers: false,
            include_repos: true,
        };

        let results = self.store.search(&query, &query_embedding);

        // Extract evidence
        let related_docs: Vec<Evidence> = results
            .chunks
            .iter()
            .take(5)
            .map(|c| Evidence {
                source: c.chunk.document_id.clone(),
                excerpt: c.chunk.content.chars().take(200).collect(),
                relevance: c.score,
            })
            .collect();

        // Generate guidance
        let steps = self.generate_steps(specification, target_backend);
        let pitfalls = self.get_pitfalls(target_backend);
        let best_practices = self.get_best_practices(target_backend);

        Ok(CompilationGuidance {
            input_summary: specification.chars().take(100).collect(),
            target_backend,
            steps,
            pitfalls,
            best_practices,
            related_docs,
        })
    }

    /// Generate compilation steps for a backend
    ///
    /// First tries to get steps from ToolKnowledgeStore using tactics and installation info,
    /// then falls back to hardcoded steps.
    pub fn generate_steps(&self, _specification: &str, backend: BackendId) -> Vec<CompilationStep> {
        // First try ToolKnowledgeStore if available
        if let Some(steps) = self.get_tool_store_steps(backend) {
            if !steps.is_empty() {
                return steps;
            }
        }

        // Fall back to hardcoded steps
        match backend {
            BackendId::Lean4 => vec![
                CompilationStep {
                    step_number: 1,
                    description: "Define types and structures".to_string(),
                    code_example: Some("structure State where\n  x : Nat\n  y : Int".to_string()),
                    verification: Some("Check that types compile without errors".to_string()),
                },
                CompilationStep {
                    step_number: 2,
                    description: "Define predicates and invariants".to_string(),
                    code_example: Some("def Invariant (s : State) : Prop := s.x > 0".to_string()),
                    verification: Some("Verify predicate is well-typed".to_string()),
                },
                CompilationStep {
                    step_number: 3,
                    description: "State theorems to prove".to_string(),
                    code_example: Some(
                        "theorem inv_preserved : Invariant s → Invariant s' := by\n  sorry"
                            .to_string(),
                    ),
                    verification: Some("Check theorem statement compiles".to_string()),
                },
                CompilationStep {
                    step_number: 4,
                    description: "Complete proofs".to_string(),
                    code_example: None,
                    verification: Some(
                        "All theorems should have complete proofs (no sorry)".to_string(),
                    ),
                },
            ],
            BackendId::TlaPlus => vec![
                CompilationStep {
                    step_number: 1,
                    description: "Define CONSTANTS and VARIABLES".to_string(),
                    code_example: Some(
                        "CONSTANTS N\nVARIABLES x, y\nvars == <<x, y>>".to_string(),
                    ),
                    verification: Some("Module should parse without errors".to_string()),
                },
                CompilationStep {
                    step_number: 2,
                    description: "Define TypeInvariant and Init".to_string(),
                    code_example: Some(
                        "TypeInvariant == x \\in Nat /\\ y \\in Int\nInit == x = 0 /\\ y = 0"
                            .to_string(),
                    ),
                    verification: Some("Check Init => TypeInvariant".to_string()),
                },
                CompilationStep {
                    step_number: 3,
                    description: "Define Next action".to_string(),
                    code_example: Some("Next == x' = x + 1 /\\ y' = y".to_string()),
                    verification: Some("Run TLC with small state space".to_string()),
                },
                CompilationStep {
                    step_number: 4,
                    description: "Define safety/liveness properties".to_string(),
                    code_example: Some("Safety == x >= 0".to_string()),
                    verification: Some("Model check all properties".to_string()),
                },
            ],
            BackendId::Kani => vec![
                CompilationStep {
                    step_number: 1,
                    description: "Add kani dependency to Cargo.toml".to_string(),
                    code_example: Some("[dev-dependencies]\nkani-verifier = \"*\"".to_string()),
                    verification: Some("cargo build should succeed".to_string()),
                },
                CompilationStep {
                    step_number: 2,
                    description: "Write proof harnesses".to_string(),
                    code_example: Some("#[kani::proof]\nfn check_add() {\n    let x: u32 = kani::any();\n    let y: u32 = kani::any();\n    kani::assume(x < 1000 && y < 1000);\n    assert!(x + y < 2000);\n}".to_string()),
                    verification: Some("cargo kani should run harness".to_string()),
                },
                CompilationStep {
                    step_number: 3,
                    description: "Add contracts to functions".to_string(),
                    code_example: Some("#[kani::requires(n < 100)]\n#[kani::ensures(|result| *result > n)]\nfn increment(n: u32) -> u32 { n + 1 }".to_string()),
                    verification: Some("cargo kani --harness <name>".to_string()),
                },
            ],
            _ => vec![CompilationStep {
                step_number: 1,
                description: format!(
                    "Consult {:?} documentation for compilation steps",
                    backend
                ),
                code_example: None,
                verification: None,
            }],
        }
    }

    /// Get common pitfalls for a backend
    ///
    /// First tries to get pitfalls from ToolKnowledgeStore,
    /// then falls back to hardcoded pitfalls.
    pub fn get_pitfalls(&self, backend: BackendId) -> Vec<String> {
        // First try ToolKnowledgeStore if available
        let tool_store_pitfalls = self.get_tool_store_pitfalls(backend);
        if !tool_store_pitfalls.is_empty() {
            return tool_store_pitfalls;
        }

        // Fall back to hardcoded pitfalls
        match backend {
            BackendId::Lean4 => vec![
                "Forgetting to import necessary libraries".to_string(),
                "Universe level mismatches in type definitions".to_string(),
                "Using 'sorry' in production code".to_string(),
            ],
            BackendId::TlaPlus => vec![
                "Forgetting to prime variables in Next".to_string(),
                "State explosion with unbounded domains".to_string(),
                "Symmetry issues with model values".to_string(),
            ],
            BackendId::Kani => vec![
                "Unbounded loops without #[kani::unwind]".to_string(),
                "Using standard library functions not supported by Kani".to_string(),
                "Overly permissive kani::any() without assume".to_string(),
            ],
            _ => vec!["Consult documentation for common pitfalls".to_string()],
        }
    }

    /// Get best practices for a backend
    ///
    /// First tries to get best practices from ToolKnowledgeStore,
    /// then falls back to hardcoded best practices.
    pub fn get_best_practices(&self, backend: BackendId) -> Vec<String> {
        // First try ToolKnowledgeStore if available
        let tool_store_practices = self.get_tool_store_best_practices(backend);
        if !tool_store_practices.is_empty() {
            return tool_store_practices;
        }

        // Fall back to hardcoded best practices
        match backend {
            BackendId::Lean4 => vec![
                "Use mathlib for common mathematical structures".to_string(),
                "Write helper lemmas for complex proofs".to_string(),
                "Use 'have' to structure proofs clearly".to_string(),
            ],
            BackendId::TlaPlus => vec![
                "Start with small state space for debugging".to_string(),
                "Use INSTANCE for modular specifications".to_string(),
                "Define TypeInvariant separately from safety properties".to_string(),
            ],
            BackendId::Kani => vec![
                "Use kani::assume to constrain inputs".to_string(),
                "Write focused, small proof harnesses".to_string(),
                "Use stub functions for unverified code".to_string(),
            ],
            _ => vec!["Follow community style guides".to_string()],
        }
    }

    /// Get compilation steps from ToolKnowledgeStore
    ///
    /// Uses installation info for setup steps and tactics for usage steps.
    fn get_tool_store_steps(&self, backend: BackendId) -> Option<Vec<CompilationStep>> {
        let tool_store = self.tool_store?;
        let tool_id = backend_id_to_tool_id(backend);
        let tool = tool_store.get(&tool_id)?;

        let mut steps = Vec::new();
        let mut step_number = 1;

        // Step 1: Installation (if available)
        if let Some(ref installation) = tool.installation {
            if let Some(method) = installation.methods.first() {
                let description = format!("Install {} using {}", tool.name, method.method_type);
                let code_example = method.command.clone();
                steps.push(CompilationStep {
                    step_number,
                    description,
                    code_example,
                    verification: Some(
                        "Verify installation with version check command".to_string(),
                    ),
                });
                step_number += 1;
            }
        }

        // Step 2: Setup/configuration (from integration info if available)
        if let Some(ref integration) = tool.integration {
            if integration.dashprove_backend {
                steps.push(CompilationStep {
                    step_number,
                    description: format!("Configure {} for DashProve integration", tool.name),
                    code_example: integration.cli_command.clone(),
                    verification: Some("Run a simple test to verify setup".to_string()),
                });
                step_number += 1;
            }
        }

        // Step 3+: Usage steps from tactics
        let tactics = tool_store.get_tactics(&tool_id);
        for tactic in tactics.iter().take(3) {
            // Limit to 3 tactics to keep steps manageable
            steps.push(CompilationStep {
                step_number,
                description: format!("Use {} tactic: {}", tactic.name, tactic.description),
                code_example: tactic.examples.first().cloned(),
                verification: tactic.when_to_use.clone(),
            });
            step_number += 1;
        }

        if steps.is_empty() {
            None
        } else {
            Some(steps)
        }
    }

    /// Get pitfalls from ToolKnowledgeStore
    ///
    /// Uses disadvantages from comparisons section.
    fn get_tool_store_pitfalls(&self, backend: BackendId) -> Vec<String> {
        let tool_store = match self.tool_store {
            Some(store) => store,
            None => return vec![],
        };

        let tool_id = backend_id_to_tool_id(backend);
        let tool = match tool_store.get(&tool_id) {
            Some(t) => t,
            None => return vec![],
        };

        // Get pitfalls from comparisons.disadvantages
        if let Some(ref comparisons) = tool.comparisons {
            if !comparisons.disadvantages.is_empty() {
                return comparisons.disadvantages.clone();
            }
        }

        // Also check error_patterns for common issues
        if !tool.error_patterns.is_empty() {
            return tool
                .error_patterns
                .iter()
                .take(3)
                .filter_map(|p| {
                    if !p.common_causes.is_empty() {
                        Some(p.common_causes.first()?.clone())
                    } else {
                        Some(format!("Watch out for: {}", p.meaning))
                    }
                })
                .collect();
        }

        vec![]
    }

    /// Get best practices from ToolKnowledgeStore
    ///
    /// Uses advantages from comparisons section and capabilities.
    fn get_tool_store_best_practices(&self, backend: BackendId) -> Vec<String> {
        let tool_store = match self.tool_store {
            Some(store) => store,
            None => return vec![],
        };

        let tool_id = backend_id_to_tool_id(backend);
        let tool = match tool_store.get(&tool_id) {
            Some(t) => t,
            None => return vec![],
        };

        let mut practices = Vec::new();

        // Get best practices from comparisons.advantages
        if let Some(ref comparisons) = tool.comparisons {
            for advantage in comparisons.advantages.iter().take(2) {
                practices.push(format!("Leverage: {}", advantage));
            }
        }

        // Add capability-based recommendations
        for cap in tool.capabilities.iter().take(2) {
            practices.push(format!("Use {} capability effectively", cap));
        }

        // Add documentation reference if available
        if let Some(ref docs) = tool.documentation {
            if docs.tutorial.is_some() {
                practices.push("Follow the official tutorial for best practices".to_string());
            }
        }

        practices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_knowledge::{
        Comparisons, InstallMethod, InstallationInfo, IntegrationInfo, Tactic, ToolKnowledge,
    };
    use tempfile::TempDir;

    #[test]
    fn test_get_tool_store_steps_with_installation_and_tactics() {
        // Create a mock tool store with installation and tactics
        let mut tool_store = ToolKnowledgeStore::new();
        let kani_tool = ToolKnowledge {
            id: "kani".to_string(),
            name: "Kani".to_string(),
            category: "rust_formal_verification".to_string(),
            subcategory: None,
            description: "Model checker for Rust".to_string(),
            long_description: None,
            capabilities: vec!["model_checking".to_string()],
            property_types: vec![],
            input_languages: vec!["rust".to_string()],
            output_formats: vec![],
            installation: Some(InstallationInfo {
                methods: vec![InstallMethod {
                    method_type: "cargo".to_string(),
                    command: Some("cargo install --locked kani-verifier".to_string()),
                    url: None,
                }],
                dependencies: vec![],
                platforms: vec![],
            }),
            documentation: None,
            tactics: vec![Tactic {
                name: "kani::proof".to_string(),
                description: "Mark function as proof harness".to_string(),
                syntax: Some("#[kani::proof]".to_string()),
                when_to_use: Some("For each property to verify".to_string()),
                examples: vec!["#[kani::proof]\nfn verify_add() {}".to_string()],
            }],
            error_patterns: vec![],
            integration: Some(IntegrationInfo {
                dashprove_backend: true,
                usl_property_types: vec![],
                cli_command: Some("cargo kani".to_string()),
            }),
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(kani_tool);

        // Create a minimal knowledge store
        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        // Test with tool store
        let expert = CompilationGuidanceExpert::with_tool_store(&store, &embedder, &tool_store);
        let steps = expert.generate_steps("test spec", BackendId::Kani);

        // Should have installation step, integration step, and tactic step
        assert!(
            steps.len() >= 2,
            "Should have multiple steps from tool store"
        );

        // First step should be installation
        assert!(
            steps[0].description.contains("Install"),
            "First step should be installation"
        );
        assert!(
            steps[0]
                .code_example
                .as_ref()
                .unwrap()
                .contains("cargo install"),
            "Should have install command"
        );
    }

    #[test]
    fn test_get_tool_store_pitfalls_from_comparisons() {
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
            tactics: vec![],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: Some(Comparisons {
                similar_tools: vec!["Coq".to_string()],
                advantages: vec!["Fast type checking".to_string()],
                disadvantages: vec![
                    "Steep learning curve".to_string(),
                    "Smaller ecosystem than Coq".to_string(),
                ],
            }),
            metadata: None,
        };
        tool_store.add_tool(lean4_tool);

        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        let expert = CompilationGuidanceExpert::with_tool_store(&store, &embedder, &tool_store);
        let pitfalls = expert.get_pitfalls(BackendId::Lean4);

        assert!(!pitfalls.is_empty(), "Should have pitfalls from tool store");
        assert!(
            pitfalls.iter().any(|p| p.contains("Steep learning curve")),
            "Should include disadvantage from comparisons"
        );
    }

    #[test]
    fn test_get_tool_store_best_practices_from_comparisons() {
        let mut tool_store = ToolKnowledgeStore::new();
        let tla_tool = ToolKnowledge {
            id: "tlaplus".to_string(),
            name: "TLA+".to_string(),
            category: "model_checker".to_string(),
            subcategory: None,
            description: "Specification language for concurrent systems".to_string(),
            long_description: None,
            capabilities: vec!["model_checking".to_string(), "temporal_logic".to_string()],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![],
            error_patterns: vec![],
            integration: None,
            performance: None,
            comparisons: Some(Comparisons {
                similar_tools: vec!["Alloy".to_string()],
                advantages: vec![
                    "Excellent for distributed systems".to_string(),
                    "Industrial adoption".to_string(),
                ],
                disadvantages: vec![],
            }),
            metadata: None,
        };
        tool_store.add_tool(tla_tool);

        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        let expert = CompilationGuidanceExpert::with_tool_store(&store, &embedder, &tool_store);
        let practices = expert.get_best_practices(BackendId::TlaPlus);

        assert!(
            !practices.is_empty(),
            "Should have best practices from tool store"
        );
        // Should have advantages and capabilities
        assert!(
            practices.iter().any(|p| p.contains("distributed systems")),
            "Should include advantage from comparisons"
        );
    }

    #[test]
    fn test_fallback_to_hardcoded_when_no_tool_store() {
        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        // Expert without tool store
        let expert = CompilationGuidanceExpert::new(&store, &embedder);

        // Should fall back to hardcoded steps
        let steps = expert.generate_steps("test", BackendId::Lean4);
        assert!(!steps.is_empty(), "Should have hardcoded steps");
        assert!(
            steps[0].description.contains("types"),
            "First Lean4 step should be about types"
        );

        // Should fall back to hardcoded pitfalls
        let pitfalls = expert.get_pitfalls(BackendId::Kani);
        assert!(!pitfalls.is_empty(), "Should have hardcoded pitfalls");
        assert!(
            pitfalls.iter().any(|p| p.contains("unwind")),
            "Kani pitfalls should mention unwinding"
        );

        // Should fall back to hardcoded best practices
        let practices = expert.get_best_practices(BackendId::TlaPlus);
        assert!(!practices.is_empty(), "Should have hardcoded practices");
        assert!(
            practices.iter().any(|p| p.contains("INSTANCE")),
            "TLA+ practices should mention INSTANCE"
        );
    }

    #[test]
    fn test_fallback_when_tool_not_in_store() {
        let tool_store = ToolKnowledgeStore::new(); // Empty store

        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        let expert = CompilationGuidanceExpert::with_tool_store(&store, &embedder, &tool_store);

        // Should fall back to hardcoded since tool not in store
        let steps = expert.generate_steps("test", BackendId::Lean4);
        assert!(!steps.is_empty(), "Should fall back to hardcoded steps");

        let pitfalls = expert.get_pitfalls(BackendId::Lean4);
        assert!(
            !pitfalls.is_empty(),
            "Should fall back to hardcoded pitfalls"
        );
    }

    // =========================================================================
    // Mutation-killing tests for hardcoded match arms
    // =========================================================================

    /// Tests that TlaPlus generates specific hardcoded steps (kills delete match arm)
    #[test]
    fn test_tlaplus_hardcoded_steps_content() {
        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        let expert = CompilationGuidanceExpert::new(&store, &embedder);
        let steps = expert.generate_steps("test", BackendId::TlaPlus);

        // TlaPlus should have 4 specific steps
        assert_eq!(steps.len(), 4, "TlaPlus should have exactly 4 steps");

        // Step 1: CONSTANTS and VARIABLES
        assert!(
            steps[0].description.contains("CONSTANTS"),
            "Step 1 should mention CONSTANTS"
        );
        assert!(
            steps[0]
                .code_example
                .as_ref()
                .unwrap()
                .contains("VARIABLES"),
            "Step 1 should have VARIABLES code"
        );

        // Step 2: TypeInvariant and Init
        assert!(
            steps[1].description.contains("TypeInvariant"),
            "Step 2 should mention TypeInvariant"
        );

        // Step 3: Next action
        assert!(
            steps[2].description.contains("Next"),
            "Step 3 should mention Next action"
        );

        // Step 4: Properties
        assert!(
            steps[3].description.contains("properties"),
            "Step 4 should mention properties"
        );
    }

    /// Tests that Kani generates specific hardcoded steps (kills delete match arm)
    #[test]
    fn test_kani_hardcoded_steps_content() {
        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        let expert = CompilationGuidanceExpert::new(&store, &embedder);
        let steps = expert.generate_steps("test", BackendId::Kani);

        // Kani should have 3 specific steps
        assert_eq!(steps.len(), 3, "Kani should have exactly 3 steps");

        // Step 1: Cargo.toml dependency
        assert!(
            steps[0].description.contains("Cargo.toml"),
            "Step 1 should mention Cargo.toml"
        );
        assert!(
            steps[0]
                .code_example
                .as_ref()
                .unwrap()
                .contains("kani-verifier"),
            "Step 1 should have kani-verifier dependency"
        );

        // Step 2: Proof harnesses
        assert!(
            steps[1].description.contains("harnesses"),
            "Step 2 should mention proof harnesses"
        );
        assert!(
            steps[1]
                .code_example
                .as_ref()
                .unwrap()
                .contains("#[kani::proof]"),
            "Step 2 should have kani::proof attribute"
        );

        // Step 3: Contracts
        assert!(
            steps[2].description.contains("contracts"),
            "Step 3 should mention contracts"
        );
        assert!(
            steps[2]
                .code_example
                .as_ref()
                .unwrap()
                .contains("#[kani::requires"),
            "Step 3 should have kani::requires"
        );
    }

    /// Tests Lean4 hardcoded pitfalls (kills delete match arm)
    #[test]
    fn test_lean4_hardcoded_pitfalls_content() {
        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        let expert = CompilationGuidanceExpert::new(&store, &embedder);
        let pitfalls = expert.get_pitfalls(BackendId::Lean4);

        // Lean4 should have 3 specific pitfalls
        assert_eq!(pitfalls.len(), 3, "Lean4 should have exactly 3 pitfalls");

        // Check specific pitfall content
        assert!(
            pitfalls.iter().any(|p| p.contains("import")),
            "Should mention import issues"
        );
        assert!(
            pitfalls.iter().any(|p| p.contains("Universe")),
            "Should mention universe level issues"
        );
        assert!(
            pitfalls.iter().any(|p| p.contains("sorry")),
            "Should mention sorry usage"
        );
    }

    /// Tests TlaPlus hardcoded pitfalls (kills delete match arm)
    #[test]
    fn test_tlaplus_hardcoded_pitfalls_content() {
        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        let expert = CompilationGuidanceExpert::new(&store, &embedder);
        let pitfalls = expert.get_pitfalls(BackendId::TlaPlus);

        // TlaPlus should have 3 specific pitfalls
        assert_eq!(pitfalls.len(), 3, "TlaPlus should have exactly 3 pitfalls");

        // Check specific pitfall content
        assert!(
            pitfalls.iter().any(|p| p.contains("prime")),
            "Should mention priming variables"
        );
        assert!(
            pitfalls.iter().any(|p| p.contains("explosion")),
            "Should mention state explosion"
        );
        assert!(
            pitfalls.iter().any(|p| p.contains("Symmetry")),
            "Should mention symmetry issues"
        );
    }

    /// Tests Lean4 hardcoded best practices (kills delete match arm)
    #[test]
    fn test_lean4_hardcoded_best_practices_content() {
        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        let expert = CompilationGuidanceExpert::new(&store, &embedder);
        let practices = expert.get_best_practices(BackendId::Lean4);

        // Lean4 should have 3 specific best practices
        assert_eq!(practices.len(), 3, "Lean4 should have exactly 3 practices");

        // Check specific practice content
        assert!(
            practices.iter().any(|p| p.contains("mathlib")),
            "Should mention mathlib"
        );
        assert!(
            practices.iter().any(|p| p.contains("lemmas")),
            "Should mention helper lemmas"
        );
        assert!(
            practices.iter().any(|p| p.contains("have")),
            "Should mention 'have' for structuring proofs"
        );
    }

    /// Tests Kani hardcoded best practices (kills delete match arm)
    #[test]
    fn test_kani_hardcoded_best_practices_content() {
        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        let expert = CompilationGuidanceExpert::new(&store, &embedder);
        let practices = expert.get_best_practices(BackendId::Kani);

        // Kani should have 3 specific best practices
        assert_eq!(practices.len(), 3, "Kani should have exactly 3 practices");

        // Check specific practice content
        assert!(
            practices.iter().any(|p| p.contains("assume")),
            "Should mention kani::assume"
        );
        assert!(
            practices.iter().any(|p| p.contains("small")),
            "Should mention small harnesses"
        );
        assert!(
            practices.iter().any(|p| p.contains("stub")),
            "Should mention stub functions"
        );
    }

    /// Tests that step numbering increments correctly (kills += mutations)
    #[test]
    fn test_tool_store_steps_numbering() {
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
            input_languages: vec!["rust".to_string()],
            output_formats: vec![],
            installation: Some(InstallationInfo {
                methods: vec![InstallMethod {
                    method_type: "cargo".to_string(),
                    command: Some("cargo install --locked kani-verifier".to_string()),
                    url: None,
                }],
                dependencies: vec![],
                platforms: vec![],
            }),
            documentation: None,
            tactics: vec![
                Tactic {
                    name: "kani::proof".to_string(),
                    description: "Mark function as proof harness".to_string(),
                    syntax: Some("#[kani::proof]".to_string()),
                    when_to_use: Some("For each property to verify".to_string()),
                    examples: vec!["#[kani::proof]".to_string()],
                },
                Tactic {
                    name: "kani::unwind".to_string(),
                    description: "Bound loop iterations".to_string(),
                    syntax: Some("#[kani::unwind(N)]".to_string()),
                    when_to_use: Some("For loops".to_string()),
                    examples: vec!["#[kani::unwind(10)]".to_string()],
                },
            ],
            error_patterns: vec![],
            integration: Some(IntegrationInfo {
                dashprove_backend: true,
                usl_property_types: vec![],
                cli_command: Some("cargo kani".to_string()),
            }),
            performance: None,
            comparisons: None,
            metadata: None,
        };
        tool_store.add_tool(kani_tool);

        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        let expert = CompilationGuidanceExpert::with_tool_store(&store, &embedder, &tool_store);
        let steps = expert.generate_steps("test", BackendId::Kani);

        // Should have at least 4 steps: install, integration, and 2 tactics
        assert!(steps.len() >= 4, "Should have at least 4 steps");

        // Verify step numbering is sequential and correct
        for (i, step) in steps.iter().enumerate() {
            assert_eq!(
                step.step_number,
                i + 1,
                "Step {} should have step_number {}",
                i,
                i + 1
            );
        }

        // Verify exact step numbers
        assert_eq!(steps[0].step_number, 1, "First step should be 1");
        assert_eq!(steps[1].step_number, 2, "Second step should be 2");
        assert_eq!(steps[2].step_number, 3, "Third step should be 3");
        assert_eq!(steps[3].step_number, 4, "Fourth step should be 4");
    }

    /// Tests pitfall handling with empty error_patterns (kills delete ! mutation)
    #[test]
    fn test_pitfalls_with_empty_error_patterns() {
        use crate::tool_knowledge::ErrorPattern;

        let mut tool_store = ToolKnowledgeStore::new();
        let tool = ToolKnowledge {
            id: "lean4".to_string(),
            name: "Lean 4".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "Theorem prover".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![],
            error_patterns: vec![
                ErrorPattern {
                    pattern: "type mismatch".to_string(),
                    meaning: "Type mismatch error".to_string(),
                    common_causes: vec![
                        "Incorrect type annotation".to_string(),
                        "Missing coercion".to_string(),
                    ],
                    fixes: vec![],
                },
                ErrorPattern {
                    pattern: "unknown identifier".to_string(),
                    meaning: "Unknown identifier error".to_string(),
                    common_causes: vec![], // Empty common_causes
                    fixes: vec![],
                },
            ],
            integration: None,
            performance: None,
            comparisons: None, // No comparisons, will fall through to error_patterns
            metadata: None,
        };
        tool_store.add_tool(tool);

        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        let expert = CompilationGuidanceExpert::with_tool_store(&store, &embedder, &tool_store);
        let pitfalls = expert.get_pitfalls(BackendId::Lean4);

        // Should get pitfalls from error_patterns
        assert!(
            !pitfalls.is_empty(),
            "Should have pitfalls from error_patterns"
        );

        // First error pattern has common_causes, second doesn't
        // Should include first cause and fallback message for second
        assert!(
            pitfalls
                .iter()
                .any(|p| p.contains("Incorrect type annotation")),
            "Should include first cause from first pattern"
        );
        assert!(
            pitfalls
                .iter()
                .any(|p| p.contains("Watch out for: Unknown identifier")),
            "Should include fallback message for pattern without causes"
        );
    }

    /// Tests that empty error_patterns properly triggers empty check
    #[test]
    fn test_pitfalls_with_empty_disadvantages_and_patterns() {
        let mut tool_store = ToolKnowledgeStore::new();
        let tool = ToolKnowledge {
            id: "lean4".to_string(),
            name: "Lean 4".to_string(),
            category: "theorem_prover".to_string(),
            subcategory: None,
            description: "Theorem prover".to_string(),
            long_description: None,
            capabilities: vec![],
            property_types: vec![],
            input_languages: vec![],
            output_formats: vec![],
            installation: None,
            documentation: None,
            tactics: vec![],
            error_patterns: vec![], // Empty error patterns
            integration: None,
            performance: None,
            comparisons: Some(Comparisons {
                similar_tools: vec![],
                advantages: vec![],
                disadvantages: vec![], // Empty disadvantages
            }),
            metadata: None,
        };
        tool_store.add_tool(tool);

        let temp_dir = TempDir::new().unwrap();
        let store = crate::store::KnowledgeStore::new(temp_dir.path().to_path_buf(), 384);
        let embedder = crate::embedding::Embedder::new(crate::EmbeddingModel::SentenceTransformers);

        let expert = CompilationGuidanceExpert::with_tool_store(&store, &embedder, &tool_store);
        let pitfalls = expert.get_pitfalls(BackendId::Lean4);

        // Should fall through and return empty vec from tool store
        // Then fall back to hardcoded pitfalls
        assert!(
            !pitfalls.is_empty(),
            "Should fall back to hardcoded pitfalls when both empty"
        );
        assert!(
            pitfalls.iter().any(|p| p.contains("sorry")),
            "Should have hardcoded Lean4 pitfalls"
        );
    }
}
