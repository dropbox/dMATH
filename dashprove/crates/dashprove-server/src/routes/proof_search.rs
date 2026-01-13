//! Proof search route handler for DashProve server
//!
//! Provides the POST /proof-search endpoint for AI-driven iterative proof search
//! with tactic learning and cross-backend hint propagation.

use async_trait::async_trait;
use axum::{extract::State, http::StatusCode, Json};
use dashprove_ai::{
    llm::{try_create_default_client, LlmClient, LlmError, LlmMessage, LlmResponse},
    ProofSearchAgent, ProofSearchConfig, ProofSearchRequest as AiProofSearchRequest,
    ProofSynthesizer,
};
use dashprove_usl::{parse, typecheck};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::types::{
    ErrorResponse, ProofResult, ProofSearchRequest, ProofSearchResponse, PropagatedHint, SearchStep,
};
use super::AppState;

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

/// POST /proof-search - AI-driven iterative proof search
///
/// Takes a property, target backend, and search parameters, and runs the
/// ProofSearchAgent to find a valid proof using learned tactic policies.
pub async fn proof_search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ProofSearchRequest>,
) -> Result<Json<ProofSearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Parse the property
    let property = parse_single_property(&req.property)?;

    // Convert backend IDs
    let backend = req.backend.clone().into();
    let propagate_to: Vec<_> = req.propagate_to.iter().cloned().map(Into::into).collect();

    // Create the proof search agent
    let llm_client: Box<dyn LlmClient> = match try_create_default_client() {
        Some(client) => client,
        None => Box::new(MockLlmClient::new(vec![
            "```lean\ntheorem example : True := by trivial\n```".to_string(),
            "```lean\ntheorem example : True := by\n  simp\n```".to_string(),
        ])),
    };

    let synthesizer = ProofSynthesizer::new(llm_client);
    let search_config = ProofSearchConfig {
        max_iterations: req.max_iterations,
        max_attempts_per_iteration: 2,
        validation_threshold: req.validation_threshold,
        success_reward: 1.0,
        failure_penalty: 0.6,
        exploration_rate: 0.2,
        max_hints: 5,
        enable_decomposition: true,
        max_decomposition_depth: 3,
        decomposition_complexity_threshold: 0.7,
        induction_mode: dashprove_ai::InductionMode::default(),
    };

    let mut agent = ProofSearchAgent::new(synthesizer).with_config(search_config);

    // Get learning system from state if available
    let learning_guard = if let Some(ref learning_lock) = state.learning {
        Some(learning_lock.read().await)
    } else {
        None
    };

    let ai_request = AiProofSearchRequest {
        property: &property,
        backend,
        context: None,
        propagate_to,
        additional_hints: req.hints.clone(),
        preferred_tactics: req.preferred_tactics.clone(),
        feedback: Vec::new(),
        learning: learning_guard.as_deref(),
    };

    let result = agent.search(ai_request).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Proof search failed".to_string(),
                details: Some(format!("{:?}", e)),
            }),
        )
    })?;

    // Convert result to response
    let proof = result.best_proof.map(|p| ProofResult {
        code: p.proof,
        confidence: p.confidence,
        tactics_used: p.tactics_used,
    });

    let steps: Vec<SearchStep> = result
        .steps
        .iter()
        .enumerate()
        .map(|(i, step)| {
            // Consider the step successful if validation passed
            let succeeded = step.validation.issues.is_empty();
            SearchStep {
                iteration: i as u32,
                tactic: step.tactics.join(","),
                succeeded,
                reward: step.reward,
            }
        })
        .collect();

    let propagated_hints: Vec<PropagatedHint> = result
        .propagated_hints
        .iter()
        .map(|h| PropagatedHint {
            source: h.source_backend.into(),
            target: h.target_backend.into(),
            hint: h.hint.clone(),
            confidence: h.confidence,
        })
        .collect();

    Ok(Json(ProofSearchResponse {
        found: proof.is_some(),
        proof,
        steps,
        propagated_hints,
        policy_weights: result.policy.weights,
    }))
}

/// Parse a single property from USL source
fn parse_single_property(
    source: &str,
) -> Result<dashprove_usl::ast::Property, (StatusCode, Json<ErrorResponse>)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_property() {
        // USL uses "or" and "not" keywords, not || and !
        let source = "theorem test { forall x: Bool. x or not x }";
        let result = parse_single_property(source);
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
    }

    #[test]
    fn test_parse_single_property_bare_expr() {
        // USL uses "or" and "not" keywords, not || and !
        let source = "forall x: Bool. x or not x";
        let result = parse_single_property(source);
        assert!(result.is_ok(), "Parse error: {:?}", result.err());
    }

    #[test]
    fn test_parse_single_property_invalid() {
        let source = "invalid syntax {{{";
        let result = parse_single_property(source);
        assert!(result.is_err());
    }
}
