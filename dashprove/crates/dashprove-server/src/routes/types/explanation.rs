use dashprove_ai::{Binding, CounterexampleExplanation, ExplanationKind, TraceStep};
use serde::{Deserialize, Serialize};

use crate::routes::types::BackendIdParam;

// ============ Counterexample Explanation Types ============

/// Request to explain a counterexample
#[derive(Debug, Deserialize)]
pub struct ExplainRequest {
    /// The property that failed (USL source)
    pub property: String,
    /// The counterexample or error output from the backend
    pub counterexample: String,
    /// Which backend produced the counterexample
    pub backend: BackendIdParam,
}

/// Response with counterexample explanation
#[derive(Debug, Serialize, Deserialize)]
pub struct ExplainResponse {
    /// Type of explanation
    pub kind: ExplanationKindResponse,
    /// Short summary
    pub summary: String,
    /// Detailed explanation
    pub details: String,
    /// Variable bindings from the counterexample
    pub bindings: Vec<BindingResponse>,
    /// Trace steps (for temporal properties)
    pub trace: Vec<TraceStepResponse>,
    /// Suggestions to fix the issue
    pub suggestions: Vec<String>,
}

/// Explanation kind for response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExplanationKindResponse {
    /// Counterexample shows specific variable values
    VariableAssignment,
    /// Counterexample is a state trace
    StateTrace,
    /// Counterexample shows a missing case
    MissingCase,
    /// Precondition was violated
    PreconditionViolation,
    /// Postcondition was not established
    PostconditionViolation,
    /// General explanation
    General,
}

impl From<ExplanationKind> for ExplanationKindResponse {
    fn from(kind: ExplanationKind) -> Self {
        match kind {
            ExplanationKind::VariableAssignment => ExplanationKindResponse::VariableAssignment,
            ExplanationKind::StateTrace => ExplanationKindResponse::StateTrace,
            ExplanationKind::MissingCase => ExplanationKindResponse::MissingCase,
            ExplanationKind::PreconditionViolation => {
                ExplanationKindResponse::PreconditionViolation
            }
            ExplanationKind::PostconditionViolation => {
                ExplanationKindResponse::PostconditionViolation
            }
            ExplanationKind::General => ExplanationKindResponse::General,
        }
    }
}

/// A variable binding in the response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingResponse {
    /// Variable name
    pub name: String,
    /// Variable value
    pub value: String,
    /// Optional type annotation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ty: Option<String>,
}

impl From<Binding> for BindingResponse {
    fn from(b: Binding) -> Self {
        BindingResponse {
            name: b.name,
            value: b.value,
            ty: b.ty,
        }
    }
}

/// A trace step in the response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStepResponse {
    /// Step index in the trace (0-based)
    pub step_number: usize,
    /// Action taken at this step (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// Variable bindings at this step
    pub state: Vec<BindingResponse>,
}

impl From<TraceStep> for TraceStepResponse {
    fn from(t: TraceStep) -> Self {
        TraceStepResponse {
            step_number: t.step_number,
            action: t.action,
            state: t.state.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<CounterexampleExplanation> for ExplainResponse {
    fn from(e: CounterexampleExplanation) -> Self {
        ExplainResponse {
            kind: e.kind.into(),
            summary: e.summary,
            details: e.details,
            bindings: e.bindings.into_iter().map(Into::into).collect(),
            trace: e.trace.into_iter().map(Into::into).collect(),
            suggestions: e.suggestions,
        }
    }
}
