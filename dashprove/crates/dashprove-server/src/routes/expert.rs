//! Expert API route handlers for DashProve server
//!
//! These endpoints use the RAG knowledge system to provide intelligent
//! recommendations, explanations, and guidance.
//!
//! Endpoints:
//! - POST /expert/backend - Get backend recommendation
//! - POST /expert/error - Get error explanation
//! - POST /expert/tactic - Get tactic suggestions
//! - POST /expert/compile - Get compilation guidance

use axum::{http::StatusCode, Json};
use dashprove_backends::BackendId;
use dashprove_knowledge::{
    BackendSelectionExpert, CompilationGuidanceExpert, Embedder, EmbeddingModel,
    ErrorExplanationExpert, ExpertContext, KnowledgeStore, PropertyType as ExpertPropertyType,
    TacticSuggestionExpert,
};
use std::path::PathBuf;

use super::types::{
    ErrorResponse, ExpertBackendAlternative, ExpertBackendRequest, ExpertBackendResponse,
    ExpertCompilationStep, ExpertCompileRequest, ExpertCompileResponse, ExpertErrorRequest,
    ExpertErrorResponse, ExpertSuggestedFix, ExpertTacticRequest, ExpertTacticResponse,
    ExpertTacticSuggestion,
};

/// Get the knowledge store directory
pub(crate) fn get_knowledge_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("dashprove")
        .join("knowledge")
}

/// Parse property types from comma-separated string
pub(crate) fn parse_expert_property_types(input: &str) -> Vec<ExpertPropertyType> {
    input
        .split(',')
        .filter_map(|s| match s.trim().to_lowercase().as_str() {
            "safety" => Some(ExpertPropertyType::Safety),
            "liveness" => Some(ExpertPropertyType::Liveness),
            "temporal" => Some(ExpertPropertyType::Temporal),
            "correctness" => Some(ExpertPropertyType::Correctness),
            "probabilistic" => Some(ExpertPropertyType::Probabilistic),
            "neural" | "neuralnetwork" | "nn" => Some(ExpertPropertyType::NeuralNetwork),
            "security" | "securityprotocol" => Some(ExpertPropertyType::SecurityProtocol),
            "refinement" => Some(ExpertPropertyType::Refinement),
            "smt" => Some(ExpertPropertyType::Smt),
            _ => None,
        })
        .collect()
}

/// POST /expert/backend - Get backend recommendation
///
/// Returns a recommended verification backend based on the specification,
/// property types, and code language. Uses the RAG knowledge system to
/// provide intelligent recommendations with alternatives.
pub async fn expert_backend(
    Json(req): Json<ExpertBackendRequest>,
) -> Result<Json<ExpertBackendResponse>, (StatusCode, Json<ErrorResponse>)> {
    let knowledge_dir = get_knowledge_dir();
    let store = KnowledgeStore::new(knowledge_dir, 384);
    let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);

    // Build context
    let property_types = req
        .property_types
        .as_ref()
        .map(|s| parse_expert_property_types(s))
        .unwrap_or_default();
    let context = ExpertContext {
        specification: req.spec,
        property_types,
        code_language: req.code_lang,
        tags: req.tags.unwrap_or_default(),
        ..Default::default()
    };

    let expert = BackendSelectionExpert::new(&store, &embedder);
    let recommendation = expert.recommend(&context).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Expert recommendation failed".to_string(),
                details: Some(e.to_string()),
            }),
        )
    })?;

    Ok(Json(ExpertBackendResponse {
        backend: recommendation.backend.into(),
        confidence: recommendation.confidence,
        rationale: recommendation.rationale,
        capabilities: recommendation.relevant_capabilities,
        limitations: recommendation.limitations,
        alternatives: recommendation
            .alternatives
            .into_iter()
            .map(|a| ExpertBackendAlternative {
                backend: a.backend.into(),
                rationale: a.rationale,
                prefer_when: a.prefer_when,
                confidence: a.confidence,
            })
            .collect(),
    }))
}

/// POST /expert/error - Get error explanation
///
/// Explains a verification error message with root cause analysis
/// and suggested fixes. Can be contextualized to a specific backend.
pub async fn expert_error(
    Json(req): Json<ExpertErrorRequest>,
) -> Result<Json<ExpertErrorResponse>, (StatusCode, Json<ErrorResponse>)> {
    let knowledge_dir = get_knowledge_dir();
    let store = KnowledgeStore::new(knowledge_dir, 384);
    let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);

    let backend = req.backend.map(|b| b.into());

    let expert = ErrorExplanationExpert::new(&store, &embedder);
    let explanation = expert.explain(&req.message, backend).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Error explanation failed".to_string(),
                details: Some(e.to_string()),
            }),
        )
    })?;

    Ok(Json(ExpertErrorResponse {
        original_error: explanation.original_error,
        explanation: explanation.explanation,
        root_cause: explanation.root_cause,
        suggested_fixes: explanation
            .suggested_fixes
            .into_iter()
            .map(|f| ExpertSuggestedFix {
                description: f.description,
                code_example: f.code_example,
                confidence: f.confidence,
            })
            .collect(),
    }))
}

/// POST /expert/tactic - Get tactic suggestions
///
/// Suggests proof tactics for a given goal and backend. Useful for
/// theorem provers like Lean4, Coq, and Isabelle.
pub async fn expert_tactic(
    Json(req): Json<ExpertTacticRequest>,
) -> Result<Json<ExpertTacticResponse>, (StatusCode, Json<ErrorResponse>)> {
    let knowledge_dir = get_knowledge_dir();
    let store = KnowledgeStore::new(knowledge_dir, 384);
    let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);

    let backend: BackendId = req.backend.into();

    let expert = TacticSuggestionExpert::new(&store, &embedder);
    let suggestions = expert
        .suggest(&req.goal, backend, req.context.as_deref())
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Tactic suggestion failed".to_string(),
                    details: Some(e.to_string()),
                }),
            )
        })?;

    Ok(Json(ExpertTacticResponse {
        suggestions: suggestions
            .into_iter()
            .map(|s| ExpertTacticSuggestion {
                tactic: s.tactic,
                backend: s.backend.into(),
                when_to_use: s.when_to_use,
                expected_effect: s.expected_effect,
                example: s.example,
                confidence: s.confidence,
                alternatives: s.alternatives,
            })
            .collect(),
    }))
}

/// POST /expert/compile - Get compilation guidance
///
/// Provides step-by-step guidance for compiling a specification to
/// a target backend, including common pitfalls and best practices.
pub async fn expert_compile(
    Json(req): Json<ExpertCompileRequest>,
) -> Result<Json<ExpertCompileResponse>, (StatusCode, Json<ErrorResponse>)> {
    let knowledge_dir = get_knowledge_dir();
    let store = KnowledgeStore::new(knowledge_dir, 384);
    let embedder = Embedder::new(EmbeddingModel::SentenceTransformers);

    let backend: BackendId = req.backend.into();

    let expert = CompilationGuidanceExpert::new(&store, &embedder);
    let guidance = expert.guide(&req.spec, backend).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Compilation guidance failed".to_string(),
                details: Some(e.to_string()),
            }),
        )
    })?;

    Ok(Json(ExpertCompileResponse {
        input_summary: guidance.input_summary,
        target_backend: guidance.target_backend.into(),
        steps: guidance
            .steps
            .into_iter()
            .map(|s| ExpertCompilationStep {
                step_number: s.step_number,
                description: s.description,
                code_example: s.code_example,
                verification: s.verification,
            })
            .collect(),
        pitfalls: guidance.pitfalls,
        best_practices: guidance.best_practices,
    }))
}
