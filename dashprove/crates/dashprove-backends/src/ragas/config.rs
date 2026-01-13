//! Ragas backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Metric type for Ragas RAG evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RagasMetric {
    /// Faithfulness - measures factual consistency
    #[default]
    Faithfulness,
    /// Answer relevancy
    AnswerRelevancy,
    /// Context precision
    ContextPrecision,
    /// Context recall
    ContextRecall,
    /// Context relevancy
    ContextRelevancy,
    /// Answer correctness
    AnswerCorrectness,
    /// Answer similarity
    AnswerSimilarity,
}

impl RagasMetric {
    pub fn as_str(&self) -> &'static str {
        match self {
            RagasMetric::Faithfulness => "faithfulness",
            RagasMetric::AnswerRelevancy => "answer_relevancy",
            RagasMetric::ContextPrecision => "context_precision",
            RagasMetric::ContextRecall => "context_recall",
            RagasMetric::ContextRelevancy => "context_relevancy",
            RagasMetric::AnswerCorrectness => "answer_correctness",
            RagasMetric::AnswerSimilarity => "answer_similarity",
        }
    }
}

/// Evaluation mode for Ragas
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvaluationMode {
    /// Single metric evaluation
    #[default]
    Single,
    /// Multiple metrics
    Multi,
    /// Full RAG pipeline evaluation
    Pipeline,
}

impl EvaluationMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            EvaluationMode::Single => "single",
            EvaluationMode::Multi => "multi",
            EvaluationMode::Pipeline => "pipeline",
        }
    }
}

/// Ragas backend configuration
#[derive(Debug, Clone)]
pub struct RagasConfig {
    /// Python interpreter path
    pub python_path: Option<PathBuf>,
    /// Primary metric to evaluate
    pub metric: RagasMetric,
    /// Evaluation mode
    pub mode: EvaluationMode,
    /// Include context metrics
    pub context_metrics: bool,
    /// Include answer metrics
    pub answer_metrics: bool,
    /// Batch size for evaluation
    pub batch_size: usize,
    /// Minimum pass rate threshold (0.0 to 1.0)
    pub pass_rate_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for RagasConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            metric: RagasMetric::Faithfulness,
            mode: EvaluationMode::Single,
            context_metrics: true,
            answer_metrics: true,
            batch_size: 8,
            pass_rate_threshold: 0.75,
            timeout: Duration::from_secs(180),
        }
    }
}

impl RagasConfig {
    /// Configure for full pipeline evaluation
    pub fn pipeline() -> Self {
        Self {
            mode: EvaluationMode::Pipeline,
            context_metrics: true,
            answer_metrics: true,
            pass_rate_threshold: 0.8,
            ..Default::default()
        }
    }

    /// Configure for faithfulness evaluation
    pub fn faithfulness_strict() -> Self {
        Self {
            metric: RagasMetric::Faithfulness,
            pass_rate_threshold: 0.9,
            ..Default::default()
        }
    }

    /// Configure for context metrics only
    pub fn context_only() -> Self {
        Self {
            metric: RagasMetric::ContextPrecision,
            mode: EvaluationMode::Multi,
            context_metrics: true,
            answer_metrics: false,
            pass_rate_threshold: 0.8,
            ..Default::default()
        }
    }

    /// Configure for answer correctness
    pub fn answer_correctness() -> Self {
        Self {
            metric: RagasMetric::AnswerCorrectness,
            pass_rate_threshold: 0.85,
            ..Default::default()
        }
    }
}
