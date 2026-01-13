//! Expert command implementations
//!
//! Provides CLI commands for RAG-powered expert recommendations.

use dashprove_backends::BackendId;
use dashprove_knowledge::{
    BackendSelectionExpert, CompilationGuidanceExpert, Embedder, EmbeddingModel,
    ErrorExplanationExpert, ExpertContext, KnowledgeStore, PropertyType, TacticSuggestionExpert,
};
use std::path::PathBuf;

/// Configuration for expert backend command
pub struct ExpertBackendConfig<'a> {
    pub spec: Option<&'a str>,
    pub property_types: Option<&'a str>,
    pub code_lang: Option<&'a str>,
    pub tags: Option<&'a str>,
    pub data_dir: Option<&'a str>,
    pub format: &'a str,
}

/// Configuration for expert error command
pub struct ExpertErrorConfig<'a> {
    pub message: Option<&'a str>,
    pub file: Option<&'a str>,
    pub backend: Option<&'a str>,
    pub data_dir: Option<&'a str>,
    pub format: &'a str,
}

/// Configuration for expert tactic command
pub struct ExpertTacticConfig<'a> {
    pub goal: &'a str,
    pub backend: &'a str,
    pub context: Option<&'a str>,
    pub data_dir: Option<&'a str>,
    pub format: &'a str,
}

/// Configuration for expert compile command
pub struct ExpertCompileConfig<'a> {
    pub spec: &'a str,
    pub backend: &'a str,
    pub data_dir: Option<&'a str>,
    pub format: &'a str,
}

/// Get the knowledge store directory
fn get_knowledge_dir(data_dir: Option<&str>) -> PathBuf {
    data_dir.map(PathBuf::from).unwrap_or_else(|| {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("dashprove")
            .join("knowledge")
    })
}

/// Parse property types from comma-separated string
fn parse_property_types(input: &str) -> Vec<PropertyType> {
    input
        .split(',')
        .filter_map(|s| match s.trim().to_lowercase().as_str() {
            "safety" => Some(PropertyType::Safety),
            "liveness" => Some(PropertyType::Liveness),
            "temporal" => Some(PropertyType::Temporal),
            "correctness" => Some(PropertyType::Correctness),
            "probabilistic" => Some(PropertyType::Probabilistic),
            "neural" | "neuralnetwork" | "nn" => Some(PropertyType::NeuralNetwork),
            "security" | "securityprotocol" => Some(PropertyType::SecurityProtocol),
            "refinement" => Some(PropertyType::Refinement),
            "smt" => Some(PropertyType::Smt),
            _ => None,
        })
        .collect()
}

/// Parse backend ID from string
fn parse_backend(backend: &str) -> Option<BackendId> {
    match backend.to_lowercase().as_str() {
        "lean4" | "lean" => Some(BackendId::Lean4),
        "tlaplus" | "tla+" | "tla" => Some(BackendId::TlaPlus),
        "kani" => Some(BackendId::Kani),
        "alloy" => Some(BackendId::Alloy),
        "isabelle" => Some(BackendId::Isabelle),
        "coq" => Some(BackendId::Coq),
        "dafny" => Some(BackendId::Dafny),
        "platform_api" | "platform-api" | "platform" => Some(BackendId::PlatformApi),
        "marabou" => Some(BackendId::Marabou),
        "alphabetacrown" | "crown" => Some(BackendId::AlphaBetaCrown),
        "eran" => Some(BackendId::Eran),
        "storm" => Some(BackendId::Storm),
        "prism" => Some(BackendId::Prism),
        "tamarin" => Some(BackendId::Tamarin),
        "proverif" => Some(BackendId::ProVerif),
        "verifpal" => Some(BackendId::Verifpal),
        "verus" => Some(BackendId::Verus),
        "creusot" => Some(BackendId::Creusot),
        "prusti" => Some(BackendId::Prusti),
        "z3" => Some(BackendId::Z3),
        "cvc5" => Some(BackendId::Cvc5),
        _ => None,
    }
}

/// Run the expert backend command
pub async fn run_expert_backend(config: ExpertBackendConfig<'_>) -> Result<(), String> {
    let knowledge_dir = get_knowledge_dir(config.data_dir);
    let store = KnowledgeStore::new(knowledge_dir, 384);
    let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);

    // Build context
    let mut context = ExpertContext::default();

    // Read specification if provided
    if let Some(spec_path) = config.spec {
        context.specification = Some(
            std::fs::read_to_string(spec_path)
                .map_err(|e| format!("Failed to read spec file: {}", e))?,
        );
    }

    // Parse property types
    if let Some(prop_types) = config.property_types {
        context.property_types = parse_property_types(prop_types);
    }

    // Set code language
    context.code_language = config.code_lang.map(String::from);

    // Parse tags
    if let Some(tags) = config.tags {
        context.tags = tags.split(',').map(|s| s.trim().to_string()).collect();
    }

    let expert = BackendSelectionExpert::new(&store, &embedder);
    let recommendation = expert
        .recommend(&context)
        .await
        .map_err(|e| format!("Expert recommendation failed: {}", e))?;

    if config.format == "json" {
        let output = serde_json::json!({
            "backend": format!("{:?}", recommendation.backend),
            "confidence": recommendation.confidence,
            "rationale": recommendation.rationale,
            "capabilities": recommendation.relevant_capabilities,
            "limitations": recommendation.limitations,
            "alternatives": recommendation.alternatives.iter().map(|a| serde_json::json!({
                "backend": format!("{:?}", a.backend),
                "rationale": a.rationale,
                "prefer_when": a.prefer_when,
                "confidence": a.confidence,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        println!("Backend Recommendation");
        println!("======================\n");
        println!(
            "Recommended: {:?} (confidence: {:.1}%)",
            recommendation.backend,
            recommendation.confidence * 100.0
        );
        println!("\n{}", recommendation.rationale);

        if !recommendation.relevant_capabilities.is_empty() {
            println!("\nCapabilities:");
            for cap in &recommendation.relevant_capabilities {
                println!("  - {}", cap);
            }
        }

        if !recommendation.limitations.is_empty() {
            println!("\nLimitations:");
            for lim in &recommendation.limitations {
                println!("  - {}", lim);
            }
        }

        if !recommendation.alternatives.is_empty() {
            println!("\nAlternatives:");
            for alt in &recommendation.alternatives {
                println!(
                    "  {:?} ({:.1}%): {}",
                    alt.backend,
                    alt.confidence * 100.0,
                    alt.prefer_when
                );
            }
        }
    }

    Ok(())
}

/// Run the expert error command
pub async fn run_expert_error(config: ExpertErrorConfig<'_>) -> Result<(), String> {
    let knowledge_dir = get_knowledge_dir(config.data_dir);
    let store = KnowledgeStore::new(knowledge_dir, 384);
    let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);

    // Get error message
    let error_message = if let Some(msg) = config.message {
        msg.to_string()
    } else if let Some(file) = config.file {
        std::fs::read_to_string(file).map_err(|e| format!("Failed to read error file: {}", e))?
    } else {
        return Err("Either --message or --file must be provided".to_string());
    };

    // Parse backend
    let backend = config.backend.and_then(parse_backend);

    let expert = ErrorExplanationExpert::new(&store, &embedder);
    let explanation = expert
        .explain(&error_message, backend)
        .await
        .map_err(|e| format!("Error explanation failed: {}", e))?;

    if config.format == "json" {
        let output = serde_json::json!({
            "original_error": explanation.original_error,
            "explanation": explanation.explanation,
            "root_cause": explanation.root_cause,
            "suggested_fixes": explanation.suggested_fixes.iter().map(|f| serde_json::json!({
                "description": f.description,
                "code_example": f.code_example,
                "confidence": f.confidence,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        println!("Error Explanation");
        println!("=================\n");
        println!("Error: {}\n", explanation.original_error);
        println!("Explanation: {}\n", explanation.explanation);
        println!("Root Cause: {}\n", explanation.root_cause);

        if !explanation.suggested_fixes.is_empty() {
            println!("Suggested Fixes:");
            for fix in &explanation.suggested_fixes {
                println!("  [{:.0}%] {}", fix.confidence * 100.0, fix.description);
                if let Some(ref example) = fix.code_example {
                    println!("        Example: {}", example);
                }
            }
        }
    }

    Ok(())
}

/// Run the expert tactic command
pub async fn run_expert_tactic(config: ExpertTacticConfig<'_>) -> Result<(), String> {
    let knowledge_dir = get_knowledge_dir(config.data_dir);
    let store = KnowledgeStore::new(knowledge_dir, 384);
    let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);

    // Parse backend
    let backend = parse_backend(config.backend)
        .ok_or_else(|| format!("Unknown backend: {}", config.backend))?;

    let expert = TacticSuggestionExpert::new(&store, &embedder);
    let suggestions = expert
        .suggest(config.goal, backend, config.context)
        .await
        .map_err(|e| format!("Tactic suggestion failed: {}", e))?;

    if config.format == "json" {
        let output: Vec<_> = suggestions
            .iter()
            .map(|s| {
                serde_json::json!({
                    "tactic": s.tactic,
                    "backend": format!("{:?}", s.backend),
                    "when_to_use": s.when_to_use,
                    "expected_effect": s.expected_effect,
                    "example": s.example,
                    "confidence": s.confidence,
                    "alternatives": s.alternatives,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        println!("Tactic Suggestions for {:?}", backend);
        println!("================================\n");
        println!("Goal: {}\n", config.goal);

        if suggestions.is_empty() {
            println!("No specific tactics suggested. Try general proof search tactics.");
        } else {
            for (i, sug) in suggestions.iter().enumerate() {
                println!(
                    "{}. {} ({:.0}% confidence)",
                    i + 1,
                    sug.tactic,
                    sug.confidence * 100.0
                );
                println!("   When: {}", sug.when_to_use);
                println!("   Effect: {}", sug.expected_effect);
                if let Some(ref example) = sug.example {
                    println!("   Example: {}", example);
                }
                if !sug.alternatives.is_empty() {
                    println!("   Alternatives: {}", sug.alternatives.join(", "));
                }
                println!();
            }
        }
    }

    Ok(())
}

/// Run the expert compile command
pub async fn run_expert_compile(config: ExpertCompileConfig<'_>) -> Result<(), String> {
    let knowledge_dir = get_knowledge_dir(config.data_dir);
    let store = KnowledgeStore::new(knowledge_dir, 384);
    let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);

    // Read specification
    let spec_content = std::fs::read_to_string(config.spec)
        .map_err(|e| format!("Failed to read spec file: {}", e))?;

    // Parse backend
    let backend = parse_backend(config.backend)
        .ok_or_else(|| format!("Unknown backend: {}", config.backend))?;

    let expert = CompilationGuidanceExpert::new(&store, &embedder);
    let guidance = expert
        .guide(&spec_content, backend)
        .await
        .map_err(|e| format!("Compilation guidance failed: {}", e))?;

    if config.format == "json" {
        let output = serde_json::json!({
            "input_summary": guidance.input_summary,
            "target_backend": format!("{:?}", guidance.target_backend),
            "steps": guidance.steps.iter().map(|s| serde_json::json!({
                "step_number": s.step_number,
                "description": s.description,
                "code_example": s.code_example,
                "verification": s.verification,
            })).collect::<Vec<_>>(),
            "pitfalls": guidance.pitfalls,
            "best_practices": guidance.best_practices,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        println!("Compilation Guidance for {:?}", guidance.target_backend);
        println!("=====================================\n");
        println!("Input: {}...\n", guidance.input_summary);

        println!("Steps:");
        for step in &guidance.steps {
            println!("\n{}. {}", step.step_number, step.description);
            if let Some(ref example) = step.code_example {
                println!(
                    "   Example:\n   ```\n   {}\n   ```",
                    example.replace('\n', "\n   ")
                );
            }
            if let Some(ref verify) = step.verification {
                println!("   Verify: {}", verify);
            }
        }

        if !guidance.pitfalls.is_empty() {
            println!("\nCommon Pitfalls:");
            for pitfall in &guidance.pitfalls {
                println!("  - {}", pitfall);
            }
        }

        if !guidance.best_practices.is_empty() {
            println!("\nBest Practices:");
            for practice in &guidance.best_practices {
                println!("  - {}", practice);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_property_types() {
        let types = parse_property_types("safety,liveness,temporal");
        assert_eq!(types.len(), 3);
        assert!(types.contains(&PropertyType::Safety));
        assert!(types.contains(&PropertyType::Liveness));
        assert!(types.contains(&PropertyType::Temporal));
    }

    #[test]
    fn test_parse_property_types_with_aliases() {
        let types = parse_property_types("nn,security");
        assert_eq!(types.len(), 2);
        assert!(types.contains(&PropertyType::NeuralNetwork));
        assert!(types.contains(&PropertyType::SecurityProtocol));
    }

    #[test]
    fn test_parse_backend() {
        assert_eq!(parse_backend("lean4"), Some(BackendId::Lean4));
        assert_eq!(parse_backend("Lean"), Some(BackendId::Lean4));
        assert_eq!(parse_backend("tla+"), Some(BackendId::TlaPlus));
        assert_eq!(parse_backend("tlaplus"), Some(BackendId::TlaPlus));
        assert_eq!(parse_backend("kani"), Some(BackendId::Kani));
        assert_eq!(parse_backend("platform_api"), Some(BackendId::PlatformApi));
        assert_eq!(parse_backend("unknown"), None);
    }

    #[test]
    fn test_get_knowledge_dir_with_custom() {
        let dir = get_knowledge_dir(Some("/custom/path"));
        assert_eq!(dir, PathBuf::from("/custom/path"));
    }

    #[test]
    fn test_get_knowledge_dir_default() {
        let dir = get_knowledge_dir(None);
        assert!(dir.to_string_lossy().contains("knowledge"));
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_parse_property_types_never_panics(input in "\\PC*") {
            let _ = parse_property_types(&input);
        }

        #[test]
        fn test_parse_backend_never_panics(input in "\\PC*") {
            let _ = parse_backend(&input);
        }

        #[test]
        fn test_parse_property_types_known_inputs(
            types in prop::sample::select(vec![
                "safety", "liveness", "temporal", "correctness",
                "probabilistic", "neural", "security", "refinement", "smt"
            ])
        ) {
            let result = parse_property_types(types);
            prop_assert!(!result.is_empty(), "Known property type should parse: {}", types);
        }

        #[test]
        fn test_parse_backend_known_inputs(
            backend in prop::sample::select(vec![
                "lean4", "lean", "tlaplus", "tla+", "kani", "alloy",
                "coq", "isabelle", "dafny", "z3", "cvc5"
            ])
        ) {
            let result = parse_backend(backend);
            prop_assert!(result.is_some(), "Known backend should parse: {}", backend);
        }
    }
}
