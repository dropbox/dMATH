//! Assistant and explanation route handlers for DashProve server

use axum::{extract::State, http::StatusCode, Json};
use dashprove_ai::{elaborate_sketch, explain_counterexample, ProofSketch};
use dashprove_backends::BackendId;
use dashprove_usl::{ast::Property, parse, typecheck};
use std::sync::Arc;

use super::types::{
    ErrorResponse, ExplainRequest, ExplainResponse, ProofSketchResponse, SketchElaborateRequest,
    SketchElaborateResponse, SketchStepResponse, TacticSuggestRequest, TacticSuggestResponse,
};
use super::AppState;

/// POST /tactics/suggest - Get tactic suggestions
pub async fn tactics_suggest(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TacticSuggestRequest>,
) -> Result<Json<TacticSuggestResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Parse property from request
    let property = parse_single_property(&req.property)?;

    // Get suggestions from assistant
    let assistant = state.assistant.read().await;
    let suggestions = assistant.suggest_tactics(&property, req.n);

    Ok(Json(TacticSuggestResponse {
        suggestions: suggestions.into_iter().map(Into::into).collect(),
    }))
}

/// POST /sketch/elaborate - Elaborate a proof sketch
pub async fn sketch_elaborate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SketchElaborateRequest>,
) -> Result<Json<SketchElaborateResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Parse property from request
    let property = parse_single_property(&req.property)?;

    // Create and elaborate sketch
    let assistant = state.assistant.read().await;
    let sketch = assistant.create_sketch(&property, &req.hints);
    let lean_code = elaborate_sketch(&sketch).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Sketch elaboration failed".to_string(),
                details: Some(e.to_string()),
            }),
        )
    })?;

    Ok(Json(SketchElaborateResponse {
        sketch: sketch_to_response(&sketch),
        lean_code,
    }))
}

/// POST /explain - Explain a counterexample in human-readable form
///
/// Takes a property, a counterexample/error from a backend, and the backend ID,
/// and returns a structured explanation with suggestions for fixing the issue.
pub async fn explain(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<ExplainRequest>,
) -> Result<Json<ExplainResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Parse the property
    let property = parse_single_property(&req.property)?;

    // Convert backend ID
    let backend: BackendId = req.backend.into();

    // Generate explanation
    let explanation = explain_counterexample(&property, &req.counterexample, &backend);

    Ok(Json(explanation.into()))
}

/// Parse a single property from USL source
fn parse_single_property(source: &str) -> Result<Property, (StatusCode, Json<ErrorResponse>)> {
    // Try wrapping in theorem if bare expression
    let wrapped = if source.trim().starts_with("theorem")
        || source.trim().starts_with("invariant")
        || source.trim().starts_with("temporal")
        || source.trim().starts_with("contract")
    {
        source.to_string()
    } else {
        format!("theorem _query {{ {} }}", source)
    };

    let spec = parse(&wrapped).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Parse error".to_string(),
                details: Some(e.to_string()),
            }),
        )
    })?;

    let typed = typecheck(spec).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Type error".to_string(),
                details: Some(e.to_string()),
            }),
        )
    })?;

    typed.spec.properties.into_iter().next().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "No property found".to_string(),
                details: None,
            }),
        )
    })
}

fn sketch_to_response(sketch: &ProofSketch) -> ProofSketchResponse {
    ProofSketchResponse {
        property_name: sketch.property_name.clone(),
        steps: sketch
            .steps
            .iter()
            .map(|s| SketchStepResponse {
                description: s.goal.clone(),
                tactic: Some(s.strategy.clone()),
                completed: s.complete,
            })
            .collect(),
        is_complete: sketch.is_complete(),
    }
}
