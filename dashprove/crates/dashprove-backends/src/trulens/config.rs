//! TruLens backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Feedback function type for evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeedbackType {
    /// Answer relevance to question
    #[default]
    AnswerRelevance,
    /// Context relevance to question
    ContextRelevance,
    /// Groundedness of answer in context
    Groundedness,
    /// Coherence of response
    Coherence,
    /// Helpfulness of response
    Helpfulness,
    /// Custom feedback function
    Custom,
}

impl FeedbackType {
    pub fn as_str(&self) -> &'static str {
        match self {
            FeedbackType::AnswerRelevance => "answer_relevance",
            FeedbackType::ContextRelevance => "context_relevance",
            FeedbackType::Groundedness => "groundedness",
            FeedbackType::Coherence => "coherence",
            FeedbackType::Helpfulness => "helpfulness",
            FeedbackType::Custom => "custom",
        }
    }
}

/// Provider for feedback evaluations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeedbackProvider {
    /// OpenAI-based feedback
    #[default]
    OpenAI,
    /// Hugging Face models
    HuggingFace,
    /// Local embeddings
    Local,
}

impl FeedbackProvider {
    pub fn as_str(&self) -> &'static str {
        match self {
            FeedbackProvider::OpenAI => "openai",
            FeedbackProvider::HuggingFace => "huggingface",
            FeedbackProvider::Local => "local",
        }
    }
}

/// TruLens backend configuration
#[derive(Debug, Clone)]
pub struct TruLensConfig {
    /// Python interpreter path
    pub python_path: Option<PathBuf>,
    /// Primary feedback type
    pub feedback_type: FeedbackType,
    /// Feedback provider
    pub provider: FeedbackProvider,
    /// Evaluate groundedness
    pub check_groundedness: bool,
    /// Evaluate relevance
    pub check_relevance: bool,
    /// Evaluate coherence
    pub check_coherence: bool,
    /// Minimum score threshold (0.0 to 1.0)
    pub score_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for TruLensConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            feedback_type: FeedbackType::AnswerRelevance,
            provider: FeedbackProvider::Local,
            check_groundedness: true,
            check_relevance: true,
            check_coherence: false,
            score_threshold: 0.7,
            timeout: Duration::from_secs(180),
        }
    }
}

impl TruLensConfig {
    /// Configure for RAG evaluation
    pub fn rag_eval() -> Self {
        Self {
            feedback_type: FeedbackType::Groundedness,
            check_groundedness: true,
            check_relevance: true,
            check_coherence: true,
            score_threshold: 0.75,
            ..Default::default()
        }
    }

    /// Configure for answer quality
    pub fn answer_quality() -> Self {
        Self {
            feedback_type: FeedbackType::AnswerRelevance,
            check_relevance: true,
            check_coherence: true,
            check_groundedness: false,
            score_threshold: 0.8,
            ..Default::default()
        }
    }

    /// Configure for strict evaluation
    pub fn strict() -> Self {
        Self {
            check_groundedness: true,
            check_relevance: true,
            check_coherence: true,
            score_threshold: 0.9,
            ..Default::default()
        }
    }
}
