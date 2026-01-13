//! Proof search command implementation
//!
//! Uses the Phase 18 ProofSearchAgent to iteratively search for proofs
//! with AI-driven tactic selection and cross-backend hint propagation.

use crate::commands::common::default_data_dir;
use async_trait::async_trait;
use dashprove::{
    learning::ProofLearningSystem,
    usl::{parse, typecheck},
};
use dashprove_ai::{
    llm::{try_create_default_client, LlmClient, LlmError, LlmMessage, LlmResponse},
    HierarchicalSearchResult, InductionMode, ProofSearchAgent, ProofSearchConfig,
    ProofSearchRequest, ProofSynthesizer,
};
use dashprove_backends::traits::BackendId;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Configuration for proof-search command
pub struct ProofSearchCmdConfig<'a> {
    /// Path to USL specification file
    pub path: &'a str,
    /// Target backend (lean4, tlaplus, kani, coq, alloy)
    pub backend: &'a str,
    /// Property name to prove (or all if not specified)
    pub property: Option<&'a str>,
    /// Maximum search iterations
    pub max_iterations: u32,
    /// Validation threshold (0.0 to 1.0)
    pub validation_threshold: f64,
    /// Additional hints to guide search
    pub hints: Vec<String>,
    /// Preferred tactics to try first
    pub tactics: Vec<String>,
    /// Backends to propagate hints to
    pub propagate_to: Vec<String>,
    /// Directory containing learning data
    pub data_dir: Option<&'a str>,
    /// Show verbose output
    pub verbose: bool,
    /// Output format (text, json)
    pub format: &'a str,
    /// Enable hierarchical decomposition
    pub hierarchical: bool,
    /// Maximum decomposition depth
    pub max_decomposition_depth: u32,
    /// Complexity threshold for decomposition
    pub decomposition_complexity_threshold: f64,
    /// Induction mode for decomposition (simple, strong, well-founded)
    pub induction_mode: &'a str,
}

/// Parse induction mode string
fn parse_induction_mode(s: &str) -> InductionMode {
    match s.to_lowercase().as_str() {
        "strong" => InductionMode::Strong,
        "well-founded" | "wf" | "wellfounded" => InductionMode::WellFounded,
        _ => InductionMode::Simple,
    }
}

/// Parse backend string to BackendId
fn parse_backend(s: &str) -> Option<BackendId> {
    match s.to_lowercase().as_str() {
        "lean" | "lean4" => Some(BackendId::Lean4),
        "tla+" | "tlaplus" | "tla" => Some(BackendId::TlaPlus),
        "kani" => Some(BackendId::Kani),
        "coq" => Some(BackendId::Coq),
        "alloy" => Some(BackendId::Alloy),
        "isabelle" => Some(BackendId::Isabelle),
        "dafny" => Some(BackendId::Dafny),
        _ => None,
    }
}

/// Mock LLM client for when no API key is configured
struct MockLlmClient {
    responses: Vec<String>,
    index: AtomicUsize,
}

impl MockLlmClient {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            index: AtomicUsize::new(0),
        }
    }

    fn next_response(&self) -> String {
        let idx = self.index.fetch_add(1, Ordering::SeqCst);
        self.responses
            .get(idx % self.responses.len())
            .cloned()
            .unwrap_or_else(|| "```lean\nsorry\n```".to_string())
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn complete(&self, _prompt: &str) -> Result<LlmResponse, LlmError> {
        Ok(LlmResponse {
            content: self.next_response(),
            model: "mock".to_string(),
            input_tokens: None,
            output_tokens: None,
            stop_reason: None,
        })
    }

    async fn complete_messages(&self, _messages: &[LlmMessage]) -> Result<LlmResponse, LlmError> {
        self.complete("").await
    }

    fn is_configured(&self) -> bool {
        true
    }

    fn model_id(&self) -> &str {
        "mock"
    }
}

/// Run proof search for a USL specification
pub async fn run_proof_search(
    config: ProofSearchCmdConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parse backend
    let backend = parse_backend(config.backend).ok_or_else(|| {
        format!(
            "Unknown backend: {}. Use: lean4, tlaplus, kani, coq, alloy, isabelle, dafny",
            config.backend
        )
    })?;

    // Parse propagate_to backends
    let propagate_to: Vec<BackendId> = config
        .propagate_to
        .iter()
        .filter_map(|s| parse_backend(s))
        .collect();

    // Read and parse specification
    let spec_path = Path::new(config.path);
    if !spec_path.exists() {
        return Err(format!("Specification file not found: {}", config.path).into());
    }

    let spec_content = std::fs::read_to_string(spec_path)?;
    let spec = parse(&spec_content).map_err(|e| format!("Parse error: {:?}", e))?;
    let typed_spec = typecheck(spec).map_err(|e| format!("Type error: {:?}", e))?;

    if typed_spec.spec.properties.is_empty() {
        println!("No properties found in specification.");
        return Ok(());
    }

    // Load learning system if available
    let data_dir = config.data_dir.map_or_else(default_data_dir, |d| d.into());
    let learning_system = ProofLearningSystem::load_from_dir(&data_dir).ok();

    // Filter properties if a specific one is requested
    let properties: Vec<_> = if let Some(prop_name) = config.property {
        typed_spec
            .spec
            .properties
            .iter()
            .filter(|p| p.name() == prop_name)
            .collect()
    } else {
        typed_spec.spec.properties.iter().collect()
    };

    if properties.is_empty() {
        if let Some(prop_name) = config.property {
            return Err(format!("Property not found: {}", prop_name).into());
        }
        println!("No properties to search.");
        return Ok(());
    }

    // Create the proof search agent
    // Try to use real LLM client if API key is configured, otherwise fall back to mock
    let llm_client: Box<dyn LlmClient> = match try_create_default_client() {
        Some(client) => {
            if config.verbose {
                println!("Using LLM client: {}", client.model_id());
            }
            client
        }
        None => {
            if config.verbose {
                println!("No LLM configured, using mock client (set ANTHROPIC_API_KEY or OPENAI_API_KEY for real synthesis)");
            }
            Box::new(MockLlmClient::new(vec![
                "```lean\ntheorem example : True := by trivial\n```".to_string(),
                "```lean\ntheorem example : True := by\n  simp\n```".to_string(),
            ]))
        }
    };
    let synthesizer = ProofSynthesizer::new(llm_client);
    let induction_mode = parse_induction_mode(config.induction_mode);
    let search_config = ProofSearchConfig {
        max_iterations: config.max_iterations,
        max_attempts_per_iteration: 2,
        validation_threshold: config.validation_threshold,
        success_reward: 1.0,
        failure_penalty: 0.6,
        exploration_rate: 0.2,
        max_hints: 5,
        enable_decomposition: config.hierarchical,
        max_decomposition_depth: config.max_decomposition_depth,
        decomposition_complexity_threshold: config.decomposition_complexity_threshold,
        induction_mode,
    };
    let mut agent = ProofSearchAgent::new(synthesizer).with_config(search_config);

    if config.verbose {
        println!("=== DashProve Proof Search ===");
        println!("File: {}", config.path);
        println!("Backend: {:?}", backend);
        println!("Properties: {}", properties.len());
        println!("Max iterations: {}", config.max_iterations);
        println!("Validation threshold: {}", config.validation_threshold);
        if config.hierarchical {
            println!(
                "Hierarchical mode: enabled (depth={}, threshold={}, induction={:?})",
                config.max_decomposition_depth,
                config.decomposition_complexity_threshold,
                induction_mode
            );
        }
        if !config.hints.is_empty() {
            println!("Hints: {:?}", config.hints);
        }
        if !config.tactics.is_empty() {
            println!("Preferred tactics: {:?}", config.tactics);
        }
        if !propagate_to.is_empty() {
            println!("Propagate to: {:?}", propagate_to);
        }
        println!();
    }

    // Unified result type that can hold either regular or hierarchical results
    let mut results: Vec<(
        String,
        dashprove_ai::ProofSearchResult,
        Option<HierarchicalSearchResult>,
    )> = Vec::new();

    for property in &properties {
        if config.verbose {
            println!("Searching for proof of: {}", property.name());
        }

        let request = ProofSearchRequest {
            property,
            backend,
            context: None,
            propagate_to: propagate_to.clone(),
            additional_hints: config.hints.clone(),
            preferred_tactics: config.tactics.clone(),
            feedback: Vec::new(),
            learning: learning_system.as_ref(),
        };

        if config.hierarchical {
            // Use hierarchical search with decomposition
            match agent.search_with_decomposition(request).await {
                Ok(hier_result) => {
                    if config.verbose && !hier_result.sub_goal_results.is_empty() {
                        println!(
                            "  Decomposed into {} sub-goals (strategy: {:?})",
                            hier_result.sub_goal_results.len(),
                            hier_result.decomposition_strategy
                        );
                        println!(
                            "  Proven sub-goals: {}/{}",
                            hier_result.proven_count(),
                            hier_result.sub_goal_results.len()
                        );
                    }
                    results.push((
                        property.name(),
                        hier_result.main_result.clone(),
                        Some(hier_result),
                    ));
                }
                Err(e) => {
                    if config.verbose {
                        eprintln!("  Search failed: {:?}", e);
                    }
                    results.push((
                        property.name(),
                        dashprove_ai::ProofSearchResult {
                            best_proof: None,
                            steps: Vec::new(),
                            propagated_hints: Vec::new(),
                            policy: dashprove_ai::TacticPolicySnapshot {
                                weights: Vec::new(),
                            },
                        },
                        None,
                    ));
                }
            }
        } else {
            // Use standard search
            match agent.search(request).await {
                Ok(result) => {
                    results.push((property.name(), result, None));
                }
                Err(e) => {
                    if config.verbose {
                        eprintln!("  Search failed: {:?}", e);
                    }
                    results.push((
                        property.name(),
                        dashprove_ai::ProofSearchResult {
                            best_proof: None,
                            steps: Vec::new(),
                            propagated_hints: Vec::new(),
                            policy: dashprove_ai::TacticPolicySnapshot {
                                weights: Vec::new(),
                            },
                        },
                        None,
                    ));
                }
            }
        }
    }

    // Output results
    if config.format == "json" {
        print_json_results(&results);
    } else {
        print_text_results(&results, config.verbose);
    }

    Ok(())
}

fn print_text_results(
    results: &[(
        String,
        dashprove_ai::ProofSearchResult,
        Option<HierarchicalSearchResult>,
    )],
    verbose: bool,
) {
    println!("\n=== Proof Search Results ===\n");

    let mut successes = 0;
    let mut failures = 0;

    for (name, result, hier_result) in results {
        let status = if result.best_proof.is_some() {
            successes += 1;
            "✓ FOUND"
        } else {
            failures += 1;
            "✗ NOT FOUND"
        };

        println!("{}: {}", name, status);

        if verbose {
            println!("  Iterations: {}", result.steps.len());

            // Show hierarchical decomposition info if available
            if let Some(hier) = hier_result {
                if hier.decomposition_strategy != dashprove_ai::DecompositionStrategy::None {
                    println!("  Decomposition: {:?}", hier.decomposition_strategy);
                    println!(
                        "  Sub-goals: {} proven / {} total",
                        hier.proven_count(),
                        hier.sub_goal_results.len()
                    );
                    println!("  Total iterations: {}", hier.total_iterations);

                    // Show sub-goal hints if any were collected
                    let hints = hier.collect_proven_hints();
                    if !hints.is_empty() {
                        println!("  Hints from sub-goals:");
                        for hint in hints.iter().take(3) {
                            println!("    - {}", hint);
                        }
                    }
                }
            }

            if let Some(ref proof) = result.best_proof {
                println!("  Confidence: {:.1}%", proof.confidence * 100.0);
                if !proof.tactics_used.is_empty() {
                    println!("  Tactics: {}", proof.tactics_used.join(", "));
                }
            }

            if !result.propagated_hints.is_empty() {
                println!("  Propagated hints: {}", result.propagated_hints.len());
                for hint in &result.propagated_hints {
                    println!(
                        "    {:?} -> {:?}: {}",
                        hint.source_backend, hint.target_backend, hint.hint
                    );
                }
            }

            if !result.policy.weights.is_empty() {
                println!("  Policy weights:");
                for (tactic, weight) in result.policy.weights.iter().take(5) {
                    println!("    {}: {:.3}", tactic, weight);
                }
            }

            println!();
        }
    }

    println!("\nSummary: {} found, {} not found", successes, failures);
}

fn print_json_results(
    results: &[(
        String,
        dashprove_ai::ProofSearchResult,
        Option<HierarchicalSearchResult>,
    )],
) {
    #[derive(serde::Serialize)]
    struct JsonResult {
        property: String,
        found: bool,
        iterations: usize,
        confidence: Option<f64>,
        tactics_used: Vec<String>,
        propagated_hints: Vec<JsonHint>,
        policy_weights: Vec<(String, f64)>,
        #[serde(skip_serializing_if = "Option::is_none")]
        hierarchical: Option<JsonHierarchical>,
    }

    #[derive(serde::Serialize)]
    struct JsonHint {
        source: String,
        target: String,
        hint: String,
        confidence: f64,
    }

    #[derive(serde::Serialize)]
    struct JsonHierarchical {
        decomposition_strategy: String,
        sub_goal_count: usize,
        proven_count: usize,
        total_iterations: u32,
        sub_goal_hints: Vec<String>,
    }

    let json_results: Vec<JsonResult> = results
        .iter()
        .map(|(name, result, hier_result)| JsonResult {
            property: name.clone(),
            found: result.best_proof.is_some(),
            iterations: result.steps.len(),
            confidence: result.best_proof.as_ref().map(|p| p.confidence),
            tactics_used: result
                .best_proof
                .as_ref()
                .map(|p| p.tactics_used.clone())
                .unwrap_or_default(),
            propagated_hints: result
                .propagated_hints
                .iter()
                .map(|h| JsonHint {
                    source: format!("{:?}", h.source_backend),
                    target: format!("{:?}", h.target_backend),
                    hint: h.hint.clone(),
                    confidence: h.confidence,
                })
                .collect(),
            policy_weights: result.policy.weights.clone(),
            hierarchical: hier_result.as_ref().and_then(|hier| {
                if hier.decomposition_strategy != dashprove_ai::DecompositionStrategy::None {
                    Some(JsonHierarchical {
                        decomposition_strategy: format!("{:?}", hier.decomposition_strategy),
                        sub_goal_count: hier.sub_goal_results.len(),
                        proven_count: hier.proven_count(),
                        total_iterations: hier.total_iterations,
                        sub_goal_hints: hier.collect_proven_hints(),
                    })
                } else {
                    None
                }
            }),
        })
        .collect();

    if let Ok(json) = serde_json::to_string_pretty(&json_results) {
        println!("{}", json);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_backend() {
        assert_eq!(parse_backend("lean4"), Some(BackendId::Lean4));
        assert_eq!(parse_backend("LEAN"), Some(BackendId::Lean4));
        assert_eq!(parse_backend("tla+"), Some(BackendId::TlaPlus));
        assert_eq!(parse_backend("TlaPlus"), Some(BackendId::TlaPlus));
        assert_eq!(parse_backend("kani"), Some(BackendId::Kani));
        assert_eq!(parse_backend("coq"), Some(BackendId::Coq));
        assert_eq!(parse_backend("unknown"), None);
    }

    #[tokio::test]
    async fn test_proof_search_missing_file() {
        let config = ProofSearchCmdConfig {
            path: "/nonexistent/file.usl",
            backend: "lean4",
            property: None,
            max_iterations: 3,
            validation_threshold: 0.5,
            hints: vec![],
            tactics: vec![],
            propagate_to: vec![],
            data_dir: None,
            verbose: false,
            format: "text",
            hierarchical: false,
            max_decomposition_depth: 3,
            decomposition_complexity_threshold: 0.7,
            induction_mode: "simple",
        };
        let result = run_proof_search(config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_proof_search_invalid_backend() {
        let config = ProofSearchCmdConfig {
            path: "test.usl",
            backend: "invalid",
            property: None,
            max_iterations: 3,
            validation_threshold: 0.5,
            hints: vec![],
            tactics: vec![],
            propagate_to: vec![],
            data_dir: None,
            verbose: false,
            format: "text",
            hierarchical: false,
            max_decomposition_depth: 3,
            decomposition_complexity_threshold: 0.7,
            induction_mode: "simple",
        };
        let result = run_proof_search(config).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown backend"));
    }

    #[test]
    fn test_parse_induction_mode() {
        assert_eq!(parse_induction_mode("simple"), InductionMode::Simple);
        assert_eq!(parse_induction_mode("strong"), InductionMode::Strong);
        assert_eq!(
            parse_induction_mode("well-founded"),
            InductionMode::WellFounded
        );
        assert_eq!(parse_induction_mode("wf"), InductionMode::WellFounded);
        assert_eq!(parse_induction_mode("unknown"), InductionMode::Simple);
    }
}
