//! LSP command execution handlers
//!
//! This module contains the command implementations for the DashProve LSP server.
//! Commands are executed in response to `workspace/executeCommand` requests.

use crate::document::DocumentStore;
use crate::symbols::property_kind_name;
use dashprove_backends::BackendId;
use dashprove_knowledge::{
    BackendRecommendation, CompilationGuidance, Embedder, ErrorExplanation, ExpertContext,
    ExpertFactory, KnowledgeStore, PropertyType, TacticSuggestion, ToolKnowledgeStore,
};
use dashprove_usl::Property;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::{Error as JsonRpcError, Result};
use tower_lsp::lsp_types::*;
use tower_lsp::Client;

// ============================================================================
// Command identifiers
// ============================================================================

/// Command identifier for verification action
pub const COMMAND_VERIFY: &str = "dashprove.verify";

/// Command identifier for showing backend information
pub const COMMAND_SHOW_BACKEND_INFO: &str = "dashprove.showBackendInfo";

/// Command identifier for RAG-backed diagnostic explanations
pub const COMMAND_EXPLAIN_DIAGNOSTIC: &str = "dashprove.explainDiagnostic";

/// Command identifier for tactic suggestions via RAG expert
pub const COMMAND_SUGGEST_TACTICS: &str = "dashprove.suggestTactics";

/// Command identifier for backend selection recommendation via RAG expert
pub const COMMAND_RECOMMEND_BACKEND: &str = "dashprove.recommendBackend";

/// Command identifier for compilation guidance via RAG expert
pub const COMMAND_COMPILATION_GUIDANCE: &str = "dashprove.compilationGuidance";

/// Command identifier for workspace-wide backend analysis via RAG expert
pub const COMMAND_ANALYZE_WORKSPACE: &str = "dashprove.analyzeWorkspace";

/// All supported commands
pub const SUPPORTED_COMMANDS: &[&str] = &[
    COMMAND_VERIFY,
    COMMAND_SHOW_BACKEND_INFO,
    COMMAND_EXPLAIN_DIAGNOSTIC,
    COMMAND_SUGGEST_TACTICS,
    COMMAND_RECOMMEND_BACKEND,
    COMMAND_COMPILATION_GUIDANCE,
    COMMAND_ANALYZE_WORKSPACE,
];

// ============================================================================
// Command execution handlers
// ============================================================================

/// Execute the verify command.
///
/// Arguments:
/// - `args[0]`: Document URI (string)
/// - `args[1]`: Property name (string)
///
/// Returns a JSON object with verification info for the client to process.
pub async fn execute_verify_command(
    client: &Client,
    documents: &DocumentStore,
    args: Vec<serde_json::Value>,
) -> Result<Option<serde_json::Value>> {
    // Parse arguments
    let (uri_str, property_name) = parse_command_args(&args)?;

    let uri = Url::parse(&uri_str)
        .map_err(|_| JsonRpcError::invalid_params(format!("Invalid URI: {}", uri_str)))?;

    // Get document and property info
    let result = documents.with_document(&uri, |doc| {
        let spec = doc.spec.as_ref()?;

        // Find the property
        let property = spec.properties.iter().find(|p| p.name() == property_name)?;

        // Determine backend and prepare verification request
        let backend = recommended_backend(property);
        let kind = property_kind_name(property);

        Some(serde_json::json!({
            "action": "verify",
            "uri": uri_str,
            "property": property_name,
            "propertyKind": kind,
            "recommendedBackend": backend,
            "specSource": doc.text.clone(),
        }))
    });

    match result.flatten() {
        Some(info) => {
            // Notify client that verification was requested
            client
                .log_message(
                    MessageType::INFO,
                    format!("Verification requested for property: {}", property_name),
                )
                .await;
            Ok(Some(info))
        }
        None => {
            client
                .log_message(
                    MessageType::ERROR,
                    format!("Property not found: {}", property_name),
                )
                .await;
            Err(JsonRpcError::invalid_params(format!(
                "Property '{}' not found in document",
                property_name
            )))
        }
    }
}

/// Execute the show backend info command.
///
/// Arguments:
/// - `args[0]`: Document URI (string)
/// - `args[1]`: Property name (string)
///
/// Shows a message with backend information for the property.
pub async fn execute_show_backend_info(
    client: &Client,
    documents: &DocumentStore,
    args: Vec<serde_json::Value>,
) -> Result<Option<serde_json::Value>> {
    // Parse arguments
    let (uri_str, property_name) = parse_command_args(&args)?;

    let uri = Url::parse(&uri_str)
        .map_err(|_| JsonRpcError::invalid_params(format!("Invalid URI: {}", uri_str)))?;

    // Get document and property info
    let result = documents.with_document(&uri, |doc| {
        let spec = doc.spec.as_ref()?;
        let property = spec.properties.iter().find(|p| p.name() == property_name)?;
        Some(backend_info_for_property(property))
    });

    match result.flatten() {
        Some(info) => {
            // Show message to user
            client.show_message(MessageType::INFO, &info).await;
            Ok(Some(serde_json::json!({
                "message": info,
                "property": property_name,
            })))
        }
        None => {
            client
                .log_message(
                    MessageType::ERROR,
                    format!("Property not found: {}", property_name),
                )
                .await;
            Err(JsonRpcError::invalid_params(format!(
                "Property '{}' not found in document",
                property_name
            )))
        }
    }
}

/// Execute the expert diagnostic explanation command.
///
/// Arguments:
/// - `args[0]`: Diagnostic message (string)
/// - `args[1]`: Optional backend name (string)
pub async fn execute_explain_diagnostic(
    client: &Client,
    store: &Arc<KnowledgeStore>,
    embedder: &Arc<Embedder>,
    tool_store: &Arc<RwLock<Option<ToolKnowledgeStore>>>,
    args: Vec<serde_json::Value>,
) -> Result<Option<serde_json::Value>> {
    let message = args
        .first()
        .and_then(|v| v.as_str())
        .ok_or_else(|| JsonRpcError::invalid_params("Expected diagnostic message"))?
        .to_string();

    let backend = args
        .get(1)
        .and_then(|v| v.as_str())
        .and_then(parse_backend_id);

    // Use ExpertFactory with ToolKnowledgeStore if available
    // Keep the guard alive for the duration of the expert call
    let tool_guard = tool_store.read().await;
    let result = {
        let expert = match tool_guard.as_ref() {
            Some(ts) => ExpertFactory::with_tool_store(store, embedder, ts).error_explanation(),
            None => ExpertFactory::new(store, embedder).error_explanation(),
        };
        expert.explain(&message, backend).await
    };
    drop(tool_guard);

    match result {
        Ok(explanation) => {
            let summary = summarize_explanation(&explanation);
            client.show_message(MessageType::INFO, summary).await;

            let fixes: Vec<serde_json::Value> = explanation
                .suggested_fixes
                .iter()
                .map(|fix| {
                    serde_json::json!({
                        "description": fix.description,
                        "codeExample": fix.code_example,
                        "confidence": fix.confidence,
                    })
                })
                .collect();

            Ok(Some(serde_json::json!({
                "action": "explainDiagnostic",
                "backend": backend.map(|b| format!("{:?}", b)),
                "message": message,
                "explanation": explanation.explanation,
                "rootCause": explanation.root_cause,
                "suggestedFixes": fixes,
            })))
        }
        Err(e) => {
            client
                .show_message(
                    MessageType::ERROR,
                    format!("Expert explanation failed: {}", e),
                )
                .await;
            Err(JsonRpcError::invalid_params(format!(
                "Expert explanation failed: {}",
                e
            )))
        }
    }
}

/// Execute the suggest tactics command using the RAG expert system.
///
/// Arguments:
/// - `args[0]`: Goal description (string)
/// - `args[1]`: Backend name (string, e.g., "lean4", "coq", "isabelle")
/// - `args[2]`: Optional context (string)
///
/// Returns a JSON object with tactic suggestions.
pub async fn execute_suggest_tactics(
    client: &Client,
    store: &Arc<KnowledgeStore>,
    embedder: &Arc<Embedder>,
    tool_store: &Arc<RwLock<Option<ToolKnowledgeStore>>>,
    args: Vec<serde_json::Value>,
) -> Result<Option<serde_json::Value>> {
    let goal = args
        .first()
        .and_then(|v| v.as_str())
        .ok_or_else(|| JsonRpcError::invalid_params("Expected goal description"))?
        .to_string();

    let backend_str = args
        .get(1)
        .and_then(|v| v.as_str())
        .ok_or_else(|| JsonRpcError::invalid_params("Expected backend name"))?;

    let backend = parse_backend_id(backend_str)
        .ok_or_else(|| JsonRpcError::invalid_params(format!("Unknown backend: {}", backend_str)))?;

    let context = args.get(2).and_then(|v| v.as_str());

    // Use ExpertFactory with ToolKnowledgeStore if available
    // Keep the guard alive for the duration of the expert call
    let tool_guard = tool_store.read().await;
    let result = {
        let expert = match tool_guard.as_ref() {
            Some(ts) => ExpertFactory::with_tool_store(store, embedder, ts).tactic_suggestion(),
            None => ExpertFactory::new(store, embedder).tactic_suggestion(),
        };
        expert.suggest(&goal, backend, context).await
    };
    drop(tool_guard);

    match result {
        Ok(suggestions) => {
            let summary = summarize_tactic_suggestions(&suggestions);
            client.show_message(MessageType::INFO, summary).await;

            let tactics: Vec<serde_json::Value> = suggestions
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "tactic": s.tactic,
                        "backend": format!("{:?}", s.backend),
                        "whenToUse": s.when_to_use,
                        "expectedEffect": s.expected_effect,
                        "example": s.example,
                        "confidence": s.confidence,
                        "alternatives": s.alternatives,
                    })
                })
                .collect();

            Ok(Some(serde_json::json!({
                "action": "suggestTactics",
                "goal": goal,
                "backend": format!("{:?}", backend),
                "suggestions": tactics,
            })))
        }
        Err(e) => {
            client
                .show_message(
                    MessageType::ERROR,
                    format!("Tactic suggestion failed: {}", e),
                )
                .await;
            Err(JsonRpcError::invalid_params(format!(
                "Tactic suggestion failed: {}",
                e
            )))
        }
    }
}

/// Execute the backend recommendation command using the RAG expert system.
///
/// Arguments:
/// - `args[0]`: Specification text (string)
/// - `args[1]`: Optional property types (comma-separated: "safety,liveness,temporal")
/// - `args[2]`: Optional code language (string, e.g., "rust", "python")
/// - `args[3]`: Optional tags (comma-separated)
///
/// Returns a JSON object with backend recommendation.
pub async fn execute_recommend_backend(
    client: &Client,
    store: &Arc<KnowledgeStore>,
    embedder: &Arc<Embedder>,
    tool_store: &Arc<RwLock<Option<ToolKnowledgeStore>>>,
    args: Vec<serde_json::Value>,
) -> Result<Option<serde_json::Value>> {
    let spec = args.first().and_then(|v| v.as_str()).map(|s| s.to_string());

    let property_types: Vec<PropertyType> = args
        .get(1)
        .and_then(|v| v.as_str())
        .map(parse_property_types)
        .unwrap_or_default();

    let code_language = args.get(2).and_then(|v| v.as_str()).map(|s| s.to_string());

    let tags: Vec<String> = args
        .get(3)
        .and_then(|v| v.as_str())
        .map(|s| s.split(',').map(|t| t.trim().to_string()).collect())
        .unwrap_or_default();

    let context = ExpertContext {
        specification: spec.clone(),
        current_backend: None,
        error_messages: vec![],
        property_types,
        code_language,
        tags,
    };

    // Use ExpertFactory with ToolKnowledgeStore if available
    // Keep the guard alive for the duration of the expert call
    let tool_guard = tool_store.read().await;
    let result = {
        let expert = match tool_guard.as_ref() {
            Some(ts) => ExpertFactory::with_tool_store(store, embedder, ts).backend_selection(),
            None => ExpertFactory::new(store, embedder).backend_selection(),
        };
        expert.recommend(&context).await
    };
    drop(tool_guard);

    match result {
        Ok(recommendation) => {
            let summary = summarize_backend_recommendation(&recommendation);
            client.show_message(MessageType::INFO, summary).await;

            let alternatives: Vec<serde_json::Value> = recommendation
                .alternatives
                .iter()
                .map(|alt| {
                    serde_json::json!({
                        "backend": format!("{:?}", alt.backend),
                        "rationale": alt.rationale,
                        "preferWhen": alt.prefer_when,
                        "confidence": alt.confidence,
                    })
                })
                .collect();

            let evidence: Vec<serde_json::Value> = recommendation
                .evidence
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "source": e.source,
                        "excerpt": e.excerpt,
                        "relevance": e.relevance,
                    })
                })
                .collect();

            Ok(Some(serde_json::json!({
                "action": "recommendBackend",
                "backend": format!("{:?}", recommendation.backend),
                "confidence": recommendation.confidence,
                "rationale": recommendation.rationale,
                "capabilities": recommendation.relevant_capabilities,
                "limitations": recommendation.limitations,
                "alternatives": alternatives,
                "evidence": evidence,
            })))
        }
        Err(e) => {
            client
                .show_message(
                    MessageType::ERROR,
                    format!("Backend recommendation failed: {}", e),
                )
                .await;
            Err(JsonRpcError::invalid_params(format!(
                "Backend recommendation failed: {}",
                e
            )))
        }
    }
}

/// Execute the compilation guidance command using the RAG expert system.
///
/// Arguments:
/// - `args[0]`: Specification text (string)
/// - `args[1]`: Target backend name (string, e.g., "lean4", "kani", "tlaplus")
///
/// Returns a JSON object with compilation guidance.
pub async fn execute_compilation_guidance(
    client: &Client,
    store: &Arc<KnowledgeStore>,
    embedder: &Arc<Embedder>,
    tool_store: &Arc<RwLock<Option<ToolKnowledgeStore>>>,
    args: Vec<serde_json::Value>,
) -> Result<Option<serde_json::Value>> {
    let spec = args
        .first()
        .and_then(|v| v.as_str())
        .ok_or_else(|| JsonRpcError::invalid_params("Expected specification text"))?
        .to_string();

    let backend_str = args
        .get(1)
        .and_then(|v| v.as_str())
        .ok_or_else(|| JsonRpcError::invalid_params("Expected target backend name"))?;

    let backend = parse_backend_id(backend_str)
        .ok_or_else(|| JsonRpcError::invalid_params(format!("Unknown backend: {}", backend_str)))?;

    // Use ExpertFactory with ToolKnowledgeStore if available
    // Keep the guard alive for the duration of the expert call
    let tool_guard = tool_store.read().await;
    let result = {
        let expert = match tool_guard.as_ref() {
            Some(ts) => ExpertFactory::with_tool_store(store, embedder, ts).compilation_guidance(),
            None => ExpertFactory::new(store, embedder).compilation_guidance(),
        };
        expert.guide(&spec, backend).await
    };
    drop(tool_guard);

    match result {
        Ok(guidance) => {
            let summary = summarize_compilation_guidance(&guidance);
            client.show_message(MessageType::INFO, summary).await;

            let steps: Vec<serde_json::Value> = guidance
                .steps
                .iter()
                .map(|step| {
                    serde_json::json!({
                        "stepNumber": step.step_number,
                        "description": step.description,
                        "codeExample": step.code_example,
                        "verification": step.verification,
                    })
                })
                .collect();

            let evidence: Vec<serde_json::Value> = guidance
                .related_docs
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "source": e.source,
                        "excerpt": e.excerpt,
                        "relevance": e.relevance,
                    })
                })
                .collect();

            Ok(Some(serde_json::json!({
                "action": "compilationGuidance",
                "inputSummary": guidance.input_summary,
                "targetBackend": format!("{:?}", guidance.target_backend),
                "steps": steps,
                "pitfalls": guidance.pitfalls,
                "bestPractices": guidance.best_practices,
                "relatedDocs": evidence,
            })))
        }
        Err(e) => {
            client
                .show_message(
                    MessageType::ERROR,
                    format!("Compilation guidance failed: {}", e),
                )
                .await;
            Err(JsonRpcError::invalid_params(format!(
                "Compilation guidance failed: {}",
                e
            )))
        }
    }
}

/// Execute the workspace analysis command.
///
/// Analyzes all open documents and provides backend recommendations for each.
/// No arguments required - analyzes all currently open documents.
///
/// Returns a JSON object with analysis results for each document.
pub async fn execute_analyze_workspace(
    client: &Client,
    documents: &DocumentStore,
    store: &Arc<KnowledgeStore>,
    embedder: &Arc<Embedder>,
    tool_store: &Arc<RwLock<Option<ToolKnowledgeStore>>>,
) -> Result<Option<serde_json::Value>> {
    // Collect all document analyses first (doesn't need the tool store)
    let analyses = documents.analyze_all(|doc| {
        let uri = doc.uri.to_string();

        // Collect property types from the document
        let property_types: Vec<PropertyType> = doc
            .spec
            .as_ref()
            .map(|spec| {
                spec.properties
                    .iter()
                    .map(property_to_expert_type)
                    .collect()
            })
            .unwrap_or_default();

        // Get specification text (first 500 chars for context)
        let spec_preview: String = doc.text.chars().take(500).collect();

        WorkspaceDocumentInfo {
            uri,
            property_types,
            spec_preview,
            type_count: doc.spec.as_ref().map(|s| s.types.len()).unwrap_or(0),
            property_count: doc.spec.as_ref().map(|s| s.properties.len()).unwrap_or(0),
            has_parse_error: doc.parse_error.is_some(),
        }
    });

    if analyses.is_empty() {
        client
            .show_message(MessageType::INFO, "No USL documents open to analyze")
            .await;
        return Ok(Some(serde_json::json!({
            "action": "analyzeWorkspace",
            "documents": [],
            "summary": "No documents open",
        })));
    }

    // Analyze each document - use the tool store for all recommendations
    // Keep the lock for the entire loop
    let tool_guard = tool_store.read().await;
    let (results, backend_counts) = {
        let expert = match tool_guard.as_ref() {
            Some(ts) => ExpertFactory::with_tool_store(store, embedder, ts).backend_selection(),
            None => ExpertFactory::new(store, embedder).backend_selection(),
        };

        let mut results = Vec::new();
        let mut backend_counts: HashMap<String, usize> = HashMap::new();

        for info in &analyses {
            // Skip documents with parse errors
            if info.has_parse_error {
                results.push(serde_json::json!({
                    "uri": info.uri,
                    "error": "Document has parse errors",
                    "typeCount": info.type_count,
                    "propertyCount": info.property_count,
                }));
                continue;
            }

            // Build expert context for this document
            let context = ExpertContext {
                specification: Some(info.spec_preview.clone()),
                current_backend: None,
                error_messages: vec![],
                property_types: info.property_types.clone(),
                code_language: None,
                tags: vec![],
            };

            // Get recommendation
            match expert.recommend(&context).await {
                Ok(recommendation) => {
                    let backend_str = format!("{:?}", recommendation.backend);
                    *backend_counts.entry(backend_str.clone()).or_insert(0) += 1;

                    results.push(serde_json::json!({
                        "uri": info.uri,
                        "recommendedBackend": backend_str,
                        "confidence": recommendation.confidence,
                        "rationale": recommendation.rationale,
                        "capabilities": recommendation.relevant_capabilities,
                        "limitations": recommendation.limitations,
                        "typeCount": info.type_count,
                        "propertyCount": info.property_count,
                        "propertyTypes": info.property_types.iter()
                            .map(|pt| format!("{:?}", pt))
                            .collect::<Vec<_>>(),
                    }));
                }
                Err(e) => {
                    results.push(serde_json::json!({
                        "uri": info.uri,
                        "error": format!("Analysis failed: {}", e),
                        "typeCount": info.type_count,
                        "propertyCount": info.property_count,
                    }));
                }
            }
        }

        (results, backend_counts)
    };
    drop(tool_guard);

    // Build summary
    let total_docs = analyses.len();
    let docs_with_errors = analyses.iter().filter(|a| a.has_parse_error).count();
    let total_types: usize = analyses.iter().map(|a| a.type_count).sum();
    let total_properties: usize = analyses.iter().map(|a| a.property_count).sum();

    let summary = format!(
        "Analyzed {} document(s): {} type(s), {} property(properties), {} with errors. Top backends: {}",
        total_docs,
        total_types,
        total_properties,
        docs_with_errors,
        format_backend_counts(&backend_counts),
    );

    client.show_message(MessageType::INFO, &summary).await;

    Ok(Some(serde_json::json!({
        "action": "analyzeWorkspace",
        "documents": results,
        "summary": {
            "totalDocuments": total_docs,
            "documentsWithErrors": docs_with_errors,
            "totalTypes": total_types,
            "totalProperties": total_properties,
            "backendDistribution": backend_counts,
        },
    })))
}

// ============================================================================
// Helper types and functions
// ============================================================================

/// Information about a document for workspace analysis
struct WorkspaceDocumentInfo {
    uri: String,
    property_types: Vec<PropertyType>,
    spec_preview: String,
    type_count: usize,
    property_count: usize,
    has_parse_error: bool,
}

/// Parse command arguments (uri, property_name).
fn parse_command_args(args: &[serde_json::Value]) -> Result<(String, String)> {
    if args.len() < 2 {
        return Err(JsonRpcError::invalid_params(
            "Expected 2 arguments: [uri, property_name]",
        ));
    }

    let uri_str = args[0]
        .as_str()
        .ok_or_else(|| JsonRpcError::invalid_params("First argument must be a string URI"))?
        .to_string();

    let property_name = args[1]
        .as_str()
        .ok_or_else(|| JsonRpcError::invalid_params("Second argument must be a property name"))?
        .to_string();

    Ok((uri_str, property_name))
}

/// Parse a backend identifier from string (matches CLI/server expert helpers).
pub fn parse_backend_id(name: &str) -> Option<BackendId> {
    match name.to_lowercase().as_str() {
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

/// Convert USL property to expert PropertyType
fn property_to_expert_type(prop: &Property) -> PropertyType {
    match prop {
        Property::Theorem(_) => PropertyType::Correctness,
        Property::Temporal(_) => PropertyType::Temporal,
        Property::Contract(_) => PropertyType::Safety,
        Property::Invariant(_) => PropertyType::Safety,
        Property::Refinement(_) => PropertyType::Refinement,
        Property::Probabilistic(_) => PropertyType::Probabilistic,
        Property::Security(_) => PropertyType::SecurityProtocol,
        Property::Semantic(_) => PropertyType::Correctness,
        Property::PlatformApi(_) => PropertyType::Safety, // Platform constraints are safety properties
        Property::Bisimulation(_) => PropertyType::Refinement, // Bisimulation is a form of refinement
        Property::Version(_) => PropertyType::Refinement, // Version improvement is a refinement relation
        Property::Capability(_) => PropertyType::Safety,  // Capability specs are safety properties
        Property::DistributedInvariant(_) => PropertyType::Safety, // Distributed invariants are safety properties
        Property::DistributedTemporal(_) => PropertyType::Temporal, // Distributed temporal properties
        Property::Composed(_) => PropertyType::Correctness, // Composed theorems are correctness properties
        Property::ImprovementProposal(_) => PropertyType::Correctness, // Improvement proposals are correctness properties
        Property::VerificationGate(_) => PropertyType::Safety, // Verification gates are safety properties
        Property::Rollback(_) => PropertyType::Safety,         // Rollbacks are safety properties
    }
}

/// Parse property types from a comma-separated string.
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

/// Format backend counts for summary display
fn format_backend_counts(counts: &HashMap<String, usize>) -> String {
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    sorted
        .into_iter()
        .take(3)
        .map(|(backend, count)| format!("{} ({})", backend, count))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Format a concise summary of backend recommendation for LSP notifications.
fn summarize_backend_recommendation(rec: &BackendRecommendation) -> String {
    let mut summary = format!(
        "Recommended backend: {:?} ({:.0}% confidence)\n{}",
        rec.backend,
        rec.confidence * 100.0,
        rec.rationale
    );

    if !rec.alternatives.is_empty() {
        summary.push_str("\n\nAlternatives:");
        for alt in rec.alternatives.iter().take(2) {
            summary.push_str(&format!("\n- {:?}: {}", alt.backend, alt.prefer_when));
        }
    }

    summary
}

/// Format a concise summary of compilation guidance for LSP notifications.
fn summarize_compilation_guidance(guidance: &CompilationGuidance) -> String {
    let mut summary = format!("Compilation guidance for {:?}:\n", guidance.target_backend);

    for (i, step) in guidance.steps.iter().take(4).enumerate() {
        summary.push_str(&format!("{}. {}\n", i + 1, step.description));
    }

    if guidance.steps.len() > 4 {
        summary.push_str(&format!("... and {} more steps", guidance.steps.len() - 4));
    }

    if !guidance.pitfalls.is_empty() {
        summary.push_str(&format!(
            "\n\nWatch out for: {}",
            guidance.pitfalls.first().unwrap_or(&String::new())
        ));
    }

    summary
}

/// Format a concise summary of tactic suggestions for LSP notifications.
fn summarize_tactic_suggestions(suggestions: &[TacticSuggestion]) -> String {
    if suggestions.is_empty() {
        return "No tactic suggestions available for this goal.".to_string();
    }

    let mut summary = format!("Found {} tactic suggestions:\n", suggestions.len());
    for (i, s) in suggestions.iter().take(3).enumerate() {
        summary.push_str(&format!(
            "{}. {} ({:.0}%): {}\n",
            i + 1,
            s.tactic,
            s.confidence * 100.0,
            s.when_to_use
        ));
    }

    if suggestions.len() > 3 {
        summary.push_str(&format!("... and {} more", suggestions.len() - 3));
    }

    summary
}

/// Format a concise summary for LSP notifications.
fn summarize_explanation(explanation: &ErrorExplanation) -> String {
    let mut summary = format!(
        "Expert explanation: {}\nRoot cause: {}",
        explanation.explanation, explanation.root_cause
    );

    if let Some(fix) = explanation.suggested_fixes.first() {
        summary.push_str(&format!(
            "\nSuggested fix ({:.0}%): {}",
            fix.confidence * 100.0,
            fix.description
        ));
    }

    summary
}

/// Get recommended backend for a property.
pub fn recommended_backend(prop: &Property) -> &'static str {
    match prop {
        Property::Theorem(_) => "lean4",
        Property::Temporal(_) => "tlaplus",
        Property::Contract(_) => "kani",
        Property::Invariant(_) => "lean4",
        Property::Refinement(_) => "lean4",
        Property::Probabilistic(_) => "probabilistic",
        Property::Security(_) => "security",
        Property::Semantic(_) => "semantic",
        Property::PlatformApi(_) => "platform_api",
        Property::Bisimulation(_) => "bisim",
        Property::Version(_) => "lean4",
        Property::Capability(_) => "lean4",
        Property::DistributedInvariant(_) => "tlaplus",
        Property::DistributedTemporal(_) => "tlaplus",
        Property::Composed(_) => "lean4",
        Property::ImprovementProposal(_) => "lean4",
        Property::VerificationGate(_) => "lean4",
        Property::Rollback(_) => "lean4",
    }
}

/// Get detailed backend information for a property.
pub fn backend_info_for_property(prop: &Property) -> String {
    match prop {
        Property::Theorem(_) => "Theorem properties are verified using LEAN 4 or Coq.\n\n\
             LEAN 4 provides interactive theorem proving with tactics.\n\
             The proof will establish the mathematical property holds for all inputs."
            .to_string(),
        Property::Temporal(_) => {
            "Temporal properties are verified using TLA+ (TLC model checker).\n\n\
             TLC exhaustively checks that the temporal formula holds across \
             all reachable states of the system."
                .to_string()
        }
        Property::Contract(_) => "Contracts are verified using Kani (Rust model checker).\n\n\
             Kani uses bounded model checking to verify that the implementation \
             satisfies the preconditions (requires) and postconditions (ensures)."
            .to_string(),
        Property::Invariant(_) => "Invariants are verified using LEAN 4 or Alloy.\n\n\
             LEAN 4 proves the invariant holds inductively.\n\
             Alloy can find counterexamples within bounded scope."
            .to_string(),
        Property::Refinement(_) => "Refinement properties are verified using LEAN 4.\n\n\
             The proof establishes a simulation relation between the abstract \
             and concrete implementations."
            .to_string(),
        Property::Probabilistic(_) => {
            "Probabilistic properties are verified using probabilistic model checkers.\n\n\
             These tools compute exact or bounded probabilities for system behaviors."
                .to_string()
        }
        Property::Security(_) => {
            "Security properties are verified using security-focused analyzers.\n\n\
             These tools check for information flow, access control, and \
             cryptographic protocol correctness."
                .to_string()
        }
        Property::Semantic(_) => {
            "Semantic properties are verified using embedding-based similarity and semantic predicates.\n\n\
             These checks look for semantic_similarity thresholds, question-answer alignment, and \
             related semantic predicates instead of exact matches."
                .to_string()
        }
        Property::PlatformApi(_) => {
            "Platform API properties define constraints on external platform behavior.\n\n\
             The platform_api backend generates Rust static checkers that enforce \
             documented API call ordering and preconditions."
                .to_string()
        }
        Property::Bisimulation(_) => {
            "Bisimulation properties verify behavioral equivalence between implementations.\n\n\
             The bisim backend compares execution traces to ensure oracle and subject \
             produce identical observable behaviors."
                .to_string()
        }
        Property::Version(_) => {
            "Version properties specify recursive self-improvement constraints.\n\n\
             They verify that a new version improves upon a previous version \
             while preserving critical properties like soundness and termination."
                .to_string()
        }
        Property::Capability(_) => {
            "Capability properties specify what a system can do.\n\n\
             They define abilities (functions the system provides) and requirements \
             (constraints that must hold for capability preservation)."
                .to_string()
        }
        Property::DistributedInvariant(_) => {
            "Distributed invariants specify coordination constraints for multi-agent systems.\n\n\
             They are verified using TLA+ with multi-process semantics where each agent \
             is modeled as a separate process with shared state."
                .to_string()
        }
        Property::DistributedTemporal(_) => {
            "Distributed temporal properties specify temporal constraints across multiple agents.\n\n\
             They are verified using TLA+ model checking with fairness constraints to ensure \
             liveness properties hold in distributed systems."
                .to_string()
        }
        Property::Composed(_) => {
            "Composed theorems combine multiple proofs using dependency-based composition.\n\n\
             They are verified using Lean 4 where the `uses` clause specifies which other \
             properties the proof depends on, and the body provides the combined proof."
                .to_string()
        }
        Property::ImprovementProposal(_) => {
            "Improvement proposals specify changes that should improve a target property.\n\n\
             They are verified using Lean 4 to ensure the proposal satisfies improvement criteria \
             and maintains required invariants."
                .to_string()
        }
        Property::VerificationGate(_) => {
            "Verification gates define checkpoints that must pass before proceeding.\n\n\
             They are verified using Lean 4 to ensure all gate checks are satisfied \
             before allowing progression to subsequent stages."
                .to_string()
        }
        Property::Rollback(_) => {
            "Rollbacks specify conditions under which a system should revert to a previous state.\n\n\
             They are verified using Lean 4 to ensure the rollback correctly restores \
             the target state when the trigger condition is met."
                .to_string()
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tower_lsp::lsp_types::Url;

    #[test]
    fn test_parse_backend_id() {
        assert_eq!(parse_backend_id("lean4"), Some(BackendId::Lean4));
        assert_eq!(parse_backend_id("LEAN"), Some(BackendId::Lean4));
        assert_eq!(parse_backend_id("tlaplus"), Some(BackendId::TlaPlus));
        assert_eq!(parse_backend_id("TLA+"), Some(BackendId::TlaPlus));
        assert_eq!(parse_backend_id("kani"), Some(BackendId::Kani));
        assert_eq!(
            parse_backend_id("platform_api"),
            Some(BackendId::PlatformApi)
        );
        assert_eq!(parse_backend_id("unknown"), None);
    }

    #[test]
    fn test_parse_property_types() {
        let types = parse_property_types("safety,liveness,temporal");
        assert_eq!(types.len(), 3);
        assert!(types.contains(&PropertyType::Safety));
        assert!(types.contains(&PropertyType::Liveness));
        assert!(types.contains(&PropertyType::Temporal));
    }

    #[test]
    fn test_parse_property_types_case_insensitive() {
        let types = parse_property_types("SAFETY, Liveness, TEMPORAL");
        assert_eq!(types.len(), 3);
    }

    #[test]
    fn test_format_backend_counts() {
        let mut counts = HashMap::new();
        counts.insert("Lean4".to_string(), 5);
        counts.insert("Kani".to_string(), 3);
        counts.insert("TlaPlus".to_string(), 1);

        let result = format_backend_counts(&counts);
        assert!(result.contains("Lean4 (5)"));
        assert!(result.contains("Kani (3)"));
    }

    #[test]
    fn test_format_backend_counts_empty() {
        let counts = HashMap::new();
        let result = format_backend_counts(&counts);
        assert_eq!(result, "");
    }

    #[test]
    fn test_recommended_backend() {
        use crate::document::Document;

        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem t { true }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(recommended_backend(&spec.properties[0]), "lean4");

        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "temporal t { always(true) }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(recommended_backend(&spec.properties[0]), "tlaplus");

        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "contract T::f(self: Int) -> Int { requires { true } ensures { true } }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(recommended_backend(&spec.properties[0]), "kani");

        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "invariant i { true }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(recommended_backend(&spec.properties[0]), "lean4");
    }

    #[test]
    fn test_property_kind_name_integration() {
        use crate::document::Document;

        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem t { true }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(property_kind_name(&spec.properties[0]), "theorem");

        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "contract T::f(self: Int) -> Int { requires { true } ensures { true } }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(property_kind_name(&spec.properties[0]), "contract");
    }

    #[test]
    fn test_backend_info_for_property() {
        use crate::document::Document;

        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem t { true }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        let info = backend_info_for_property(&spec.properties[0]);
        assert!(info.contains("LEAN 4"));
        assert!(info.contains("Coq"));
    }

    #[test]
    fn test_parse_command_args_valid() {
        let args = vec![
            serde_json::json!("file:///test.usl"),
            serde_json::json!("my_property"),
        ];
        let result = parse_command_args(&args);
        assert!(result.is_ok());
        let (uri, name) = result.unwrap();
        assert_eq!(uri, "file:///test.usl");
        assert_eq!(name, "my_property");
    }

    #[test]
    fn test_parse_command_args_missing() {
        let args = vec![serde_json::json!("file:///test.usl")];
        let result = parse_command_args(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_command_args_wrong_type() {
        let args = vec![serde_json::json!(123), serde_json::json!("my_property")];
        let result = parse_command_args(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_command_args_second_arg_wrong_type() {
        let args = vec![
            serde_json::json!("file:///test.usl"),
            serde_json::json!(123),
        ];
        let result = parse_command_args(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_command_args_empty() {
        let args: Vec<serde_json::Value> = vec![];
        let result = parse_command_args(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_property_to_expert_type() {
        use crate::document::Document;

        // Theorem -> Correctness
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem t { true }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(
            property_to_expert_type(&spec.properties[0]),
            PropertyType::Correctness
        );

        // Temporal -> Temporal
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "temporal t { always(true) }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(
            property_to_expert_type(&spec.properties[0]),
            PropertyType::Temporal
        );

        // Contract -> Safety
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "contract T::f(self: Int) -> Int { requires { true } ensures { true } }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(
            property_to_expert_type(&spec.properties[0]),
            PropertyType::Safety
        );

        // Invariant -> Safety
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "invariant i { true }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(
            property_to_expert_type(&spec.properties[0]),
            PropertyType::Safety
        );

        // Refinement -> Refinement
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "refinement optimized refines base { abstraction { to_base(opt) == base } simulation { forall s: State . step(s) == step(s) } }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(
            property_to_expert_type(&spec.properties[0]),
            PropertyType::Refinement
        );
    }

    #[test]
    fn test_summarize_explanation() {
        let explanation = ErrorExplanation {
            original_error: "error: type mismatch".to_string(),
            explanation: "Test explanation".to_string(),
            root_cause: "Test cause".to_string(),
            suggested_fixes: vec![],
            related_docs: vec![],
            similar_issues: vec![],
        };
        let summary = summarize_explanation(&explanation);
        assert!(summary.contains("Test explanation"));
        assert!(summary.contains("Test cause"));
    }

    #[test]
    fn test_summarize_explanation_with_fix() {
        use dashprove_knowledge::SuggestedFix;

        let explanation = ErrorExplanation {
            original_error: "error: missing import".to_string(),
            explanation: "Error occurred".to_string(),
            root_cause: "Missing import".to_string(),
            suggested_fixes: vec![SuggestedFix {
                description: "Add the missing import".to_string(),
                code_example: Some("import foo".to_string()),
                confidence: 0.85,
            }],
            related_docs: vec![],
            similar_issues: vec![],
        };
        let summary = summarize_explanation(&explanation);
        assert!(summary.contains("Add the missing import"));
        assert!(summary.contains("85%"));
    }

    #[test]
    fn test_summarize_tactic_suggestions_empty() {
        let suggestions: Vec<TacticSuggestion> = vec![];
        let summary = summarize_tactic_suggestions(&suggestions);
        assert!(summary.contains("No tactic suggestions"));
    }

    #[test]
    fn test_summarize_tactic_suggestions_with_items() {
        let suggestions = vec![
            TacticSuggestion {
                tactic: "simp".to_string(),
                backend: BackendId::Lean4,
                when_to_use: "Simplify expressions".to_string(),
                expected_effect: "Reduces goal".to_string(),
                example: Some("simp [add_comm]".to_string()),
                confidence: 0.9,
                alternatives: vec![],
            },
            TacticSuggestion {
                tactic: "ring".to_string(),
                backend: BackendId::Lean4,
                when_to_use: "For ring expressions".to_string(),
                expected_effect: "Solves ring equations".to_string(),
                example: None,
                confidence: 0.8,
                alternatives: vec!["omega".to_string()],
            },
        ];
        let summary = summarize_tactic_suggestions(&suggestions);
        assert!(summary.contains("2 tactic suggestions"));
        assert!(summary.contains("simp"));
        assert!(summary.contains("90%"));
    }

    #[test]
    fn test_summarize_tactic_suggestions_truncates() {
        let suggestions = vec![
            TacticSuggestion {
                tactic: "tactic1".to_string(),
                backend: BackendId::Lean4,
                when_to_use: "Use 1".to_string(),
                expected_effect: "".to_string(),
                example: None,
                confidence: 0.9,
                alternatives: vec![],
            },
            TacticSuggestion {
                tactic: "tactic2".to_string(),
                backend: BackendId::Lean4,
                when_to_use: "Use 2".to_string(),
                expected_effect: "".to_string(),
                example: None,
                confidence: 0.8,
                alternatives: vec![],
            },
            TacticSuggestion {
                tactic: "tactic3".to_string(),
                backend: BackendId::Lean4,
                when_to_use: "Use 3".to_string(),
                expected_effect: "".to_string(),
                example: None,
                confidence: 0.7,
                alternatives: vec![],
            },
            TacticSuggestion {
                tactic: "tactic4".to_string(),
                backend: BackendId::Lean4,
                when_to_use: "Use 4".to_string(),
                expected_effect: "".to_string(),
                example: None,
                confidence: 0.6,
                alternatives: vec![],
            },
        ];
        let summary = summarize_tactic_suggestions(&suggestions);
        assert!(summary.contains("... and 1 more"));
    }

    #[test]
    fn test_summarize_backend_recommendation() {
        use dashprove_knowledge::BackendAlternative;

        let rec = BackendRecommendation {
            backend: BackendId::Lean4,
            confidence: 0.95,
            rationale: "Best for theorem proving".to_string(),
            relevant_capabilities: vec!["tactics".to_string()],
            limitations: vec![],
            alternatives: vec![BackendAlternative {
                backend: BackendId::Coq,
                rationale: "Also good".to_string(),
                prefer_when: "Need dependent types".to_string(),
                confidence: 0.8,
            }],
            evidence: vec![],
        };
        let summary = summarize_backend_recommendation(&rec);
        assert!(summary.contains("Lean4"));
        assert!(summary.contains("95%"));
        assert!(summary.contains("Best for theorem proving"));
        assert!(summary.contains("Alternatives"));
        assert!(summary.contains("Coq"));
    }

    #[test]
    fn test_summarize_backend_recommendation_no_alternatives() {
        let rec = BackendRecommendation {
            backend: BackendId::Kani,
            confidence: 0.9,
            rationale: "Best for Rust verification".to_string(),
            relevant_capabilities: vec![],
            limitations: vec![],
            alternatives: vec![],
            evidence: vec![],
        };
        let summary = summarize_backend_recommendation(&rec);
        assert!(summary.contains("Kani"));
        assert!(!summary.contains("Alternatives"));
    }

    #[test]
    fn test_summarize_compilation_guidance() {
        use dashprove_knowledge::CompilationStep;

        let guidance = CompilationGuidance {
            input_summary: "theorem test".to_string(),
            target_backend: BackendId::Lean4,
            steps: vec![
                CompilationStep {
                    step_number: 1,
                    description: "Define the lemma".to_string(),
                    code_example: None,
                    verification: None,
                },
                CompilationStep {
                    step_number: 2,
                    description: "Apply tactics".to_string(),
                    code_example: Some("apply h".to_string()),
                    verification: None,
                },
            ],
            pitfalls: vec!["Watch for universe issues".to_string()],
            best_practices: vec![],
            related_docs: vec![],
        };
        let summary = summarize_compilation_guidance(&guidance);
        assert!(summary.contains("Lean4"));
        assert!(summary.contains("Define the lemma"));
        assert!(summary.contains("Apply tactics"));
        assert!(summary.contains("universe issues"));
    }

    #[test]
    fn test_summarize_compilation_guidance_many_steps() {
        use dashprove_knowledge::CompilationStep;

        let steps: Vec<CompilationStep> = (1..=6)
            .map(|i| CompilationStep {
                step_number: i,
                description: format!("Step {}", i),
                code_example: None,
                verification: None,
            })
            .collect();

        let guidance = CompilationGuidance {
            input_summary: "".to_string(),
            target_backend: BackendId::Z3,
            steps,
            pitfalls: vec![],
            best_practices: vec![],
            related_docs: vec![],
        };
        let summary = summarize_compilation_guidance(&guidance);
        assert!(summary.contains("... and 2 more steps"));
    }

    #[test]
    fn test_summarize_compilation_guidance_empty_pitfalls() {
        let guidance = CompilationGuidance {
            input_summary: "".to_string(),
            target_backend: BackendId::Z3,
            steps: vec![],
            pitfalls: vec![],
            best_practices: vec![],
            related_docs: vec![],
        };
        let summary = summarize_compilation_guidance(&guidance);
        assert!(!summary.contains("Watch out for"));
    }

    #[test]
    fn test_format_backend_counts_limits_to_three() {
        let mut counts = HashMap::new();
        counts.insert("A".to_string(), 10);
        counts.insert("B".to_string(), 5);
        counts.insert("C".to_string(), 3);
        counts.insert("D".to_string(), 1);

        let result = format_backend_counts(&counts);
        // Should contain top 3 but not 4th
        assert!(result.contains("A (10)"));
        assert!(result.contains("B (5)"));
        assert!(result.contains("C (3)"));
        assert!(!result.contains("D (1)"));
    }

    #[test]
    fn test_recommended_backend_all_variants() {
        use crate::document::Document;

        // Refinement
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "refinement optimized refines base { abstraction { to_base(opt) == base } simulation { forall s: State . step(s) == step(s) } }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(recommended_backend(&spec.properties[0]), "lean4");

        // Probabilistic
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "probabilistic p { probability(success) == 0.5 }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(recommended_backend(&spec.properties[0]), "probabilistic");

        // Security
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "security s { not leaked(secret) }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(recommended_backend(&spec.properties[0]), "security");

        // Semantic
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "semantic_property s { similarity(x, y) > 0.9 }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(recommended_backend(&spec.properties[0]), "semantic");

        // PlatformApi
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "platform_api Metal { state Buffer { enum Status { Created } } }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(recommended_backend(&spec.properties[0]), "platform_api");

        // Bisimulation
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "bisimulation b { oracle: \"./oracle\" subject: \"./subject\" equivalent on { output } }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        assert_eq!(recommended_backend(&spec.properties[0]), "bisim");
    }

    #[test]
    fn test_backend_info_for_property_all_variants() {
        use crate::document::Document;

        // Temporal
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "temporal t { always(true) }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        let info = backend_info_for_property(&spec.properties[0]);
        assert!(info.contains("TLA+"));
        assert!(info.contains("TLC"));

        // Contract
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "contract T::f(self: Int) -> Int { requires { true } ensures { true } }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        let info = backend_info_for_property(&spec.properties[0]);
        assert!(info.contains("Kani"));

        // Invariant
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "invariant i { true }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        let info = backend_info_for_property(&spec.properties[0]);
        assert!(info.contains("LEAN 4") || info.contains("Alloy"));

        // Refinement
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "refinement optimized refines base { abstraction { to_base(opt) == base } simulation { forall s: State . step(s) == step(s) } }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        let info = backend_info_for_property(&spec.properties[0]);
        assert!(info.contains("simulation relation"));

        // Probabilistic
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "probabilistic p { probability(success) == 0.5 }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        let info = backend_info_for_property(&spec.properties[0]);
        assert!(info.contains("probabilistic"));

        // Security
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "security s { not leaked(secret) }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        let info = backend_info_for_property(&spec.properties[0]);
        assert!(info.contains("cryptographic") || info.contains("Security"));

        // Semantic
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "semantic_property s { similarity(x, y) > 0.9 }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        let info = backend_info_for_property(&spec.properties[0]);
        assert!(info.contains("embedding") || info.contains("semantic"));

        // PlatformApi
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "platform_api Metal { state Buffer { enum Status { Created } } }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        let info = backend_info_for_property(&spec.properties[0]);
        assert!(info.contains("platform"));

        // Bisimulation
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "bisimulation b { oracle: \"./oracle\" subject: \"./subject\" equivalent on { output } }".to_string(),
        );
        let spec = doc.spec.as_ref().unwrap();
        let info = backend_info_for_property(&spec.properties[0]);
        assert!(info.contains("Bisimulation") || info.contains("behavioral"));
    }

    /// Integration tests for expert commands with ToolKnowledgeStore
    mod expert_integration {
        use super::*;
        use dashprove_knowledge::{Embedder, EmbeddingModel, KnowledgeStore, ToolKnowledgeStore};
        use std::path::PathBuf;

        fn create_test_store() -> Arc<KnowledgeStore> {
            Arc::new(KnowledgeStore::new(PathBuf::from("/tmp/test_store"), 384))
        }

        fn create_test_embedder() -> Arc<Embedder> {
            Arc::new(Embedder::new(EmbeddingModel::SentenceTransformers))
        }

        fn get_tools_path() -> PathBuf {
            // Try local data directory first (for development)
            let local_tools = PathBuf::from("data/knowledge/tools");
            if local_tools.exists() {
                return local_tools;
            }
            // Try from crate root
            let crate_dir = env!("CARGO_MANIFEST_DIR");
            PathBuf::from(crate_dir)
                .parent()
                .and_then(|p| p.parent())
                .map(|p| p.join("data/knowledge/tools"))
                .unwrap_or(local_tools)
        }

        #[tokio::test]
        async fn test_expert_factory_with_none_tool_store() {
            let store = create_test_store();
            let embedder = create_test_embedder();
            let tool_store: Arc<RwLock<Option<ToolKnowledgeStore>>> = Arc::new(RwLock::new(None));

            // Should create factory without tool store
            let tool_guard = tool_store.read().await;
            let factory = match tool_guard.as_ref() {
                Some(ts) => ExpertFactory::with_tool_store(&store, &embedder, ts),
                None => ExpertFactory::new(&store, &embedder),
            };
            assert!(!factory.has_tool_store());
        }

        #[tokio::test]
        async fn test_expert_factory_with_loaded_tool_store() {
            let store = create_test_store();
            let embedder = create_test_embedder();
            let tools_path = get_tools_path();

            if !tools_path.exists() {
                eprintln!("Skipping: tools directory not found at {:?}", tools_path);
                return;
            }

            let loaded_store = ToolKnowledgeStore::load_from_dir(&tools_path)
                .await
                .expect("Failed to load tool knowledge");

            let tool_store: Arc<RwLock<Option<ToolKnowledgeStore>>> =
                Arc::new(RwLock::new(Some(loaded_store)));

            // Should create factory with tool store
            let tool_guard = tool_store.read().await;
            let factory = match tool_guard.as_ref() {
                Some(ts) => ExpertFactory::with_tool_store(&store, &embedder, ts),
                None => ExpertFactory::new(&store, &embedder),
            };
            assert!(factory.has_tool_store());
        }

        #[tokio::test]
        async fn test_tactic_expert_with_tool_store() {
            let store = create_test_store();
            let embedder = create_test_embedder();
            let tools_path = get_tools_path();

            if !tools_path.exists() {
                eprintln!("Skipping: tools directory not found at {:?}", tools_path);
                return;
            }

            let loaded_store = ToolKnowledgeStore::load_from_dir(&tools_path)
                .await
                .expect("Failed to load tool knowledge");

            let tool_store: Arc<RwLock<Option<ToolKnowledgeStore>>> =
                Arc::new(RwLock::new(Some(loaded_store)));

            let tool_guard = tool_store.read().await;
            let result = {
                let expert = match tool_guard.as_ref() {
                    Some(ts) => {
                        ExpertFactory::with_tool_store(&store, &embedder, ts).tactic_suggestion()
                    }
                    None => ExpertFactory::new(&store, &embedder).tactic_suggestion(),
                };
                expert
                    .suggest("prove basic property", BackendId::Z3, None)
                    .await
            };
            drop(tool_guard);

            // Should succeed and return suggestions
            assert!(
                result.is_ok(),
                "Tactic suggestion failed: {:?}",
                result.err()
            );
            let suggestions = result.unwrap();
            // Should have at least some suggestions from tool store or fallback
            assert!(
                !suggestions.is_empty(),
                "Expected non-empty tactic suggestions"
            );
        }

        #[tokio::test]
        async fn test_error_expert_with_tool_store() {
            let store = create_test_store();
            let embedder = create_test_embedder();
            let tools_path = get_tools_path();

            if !tools_path.exists() {
                eprintln!("Skipping: tools directory not found at {:?}", tools_path);
                return;
            }

            let loaded_store = ToolKnowledgeStore::load_from_dir(&tools_path)
                .await
                .expect("Failed to load tool knowledge");

            let tool_store: Arc<RwLock<Option<ToolKnowledgeStore>>> =
                Arc::new(RwLock::new(Some(loaded_store)));

            let tool_guard = tool_store.read().await;
            let result = {
                let expert = match tool_guard.as_ref() {
                    Some(ts) => {
                        ExpertFactory::with_tool_store(&store, &embedder, ts).error_explanation()
                    }
                    None => ExpertFactory::new(&store, &embedder).error_explanation(),
                };
                expert
                    .explain("timeout exceeded", Some(BackendId::Z3))
                    .await
            };
            drop(tool_guard);

            // Should succeed and return an explanation
            assert!(
                result.is_ok(),
                "Error explanation failed: {:?}",
                result.err()
            );
            let explanation = result.unwrap();
            assert!(!explanation.explanation.is_empty());
        }

        #[tokio::test]
        async fn test_backend_selection_expert_with_tool_store() {
            let store = create_test_store();
            let embedder = create_test_embedder();
            let tools_path = get_tools_path();

            if !tools_path.exists() {
                eprintln!("Skipping: tools directory not found at {:?}", tools_path);
                return;
            }

            let loaded_store = ToolKnowledgeStore::load_from_dir(&tools_path)
                .await
                .expect("Failed to load tool knowledge");

            let tool_store: Arc<RwLock<Option<ToolKnowledgeStore>>> =
                Arc::new(RwLock::new(Some(loaded_store)));

            let context = ExpertContext {
                specification: Some("theorem test: x + y = y + x".to_string()),
                property_types: vec![PropertyType::Correctness],
                ..Default::default()
            };

            let tool_guard = tool_store.read().await;
            let result = {
                let expert = match tool_guard.as_ref() {
                    Some(ts) => {
                        ExpertFactory::with_tool_store(&store, &embedder, ts).backend_selection()
                    }
                    None => ExpertFactory::new(&store, &embedder).backend_selection(),
                };
                expert.recommend(&context).await
            };
            drop(tool_guard);

            // Should succeed and return a recommendation
            assert!(
                result.is_ok(),
                "Backend selection failed: {:?}",
                result.err()
            );
            let recommendation = result.unwrap();
            assert!(recommendation.confidence > 0.0);
        }
    }
}
