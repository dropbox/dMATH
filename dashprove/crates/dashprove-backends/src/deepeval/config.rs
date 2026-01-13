//! DeepEval backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Metric type for DeepEval
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeepEvalMetric {
    /// Answer relevancy
    #[default]
    AnswerRelevancy,
    /// Faithfulness
    Faithfulness,
    /// Contextual precision
    ContextualPrecision,
    /// Contextual recall
    ContextualRecall,
    /// Hallucination
    Hallucination,
    /// Toxicity
    Toxicity,
    /// Bias
    Bias,
    /// G-Eval (custom criteria)
    GEval,
}

impl DeepEvalMetric {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeepEvalMetric::AnswerRelevancy => "answer_relevancy",
            DeepEvalMetric::Faithfulness => "faithfulness",
            DeepEvalMetric::ContextualPrecision => "contextual_precision",
            DeepEvalMetric::ContextualRecall => "contextual_recall",
            DeepEvalMetric::Hallucination => "hallucination",
            DeepEvalMetric::Toxicity => "toxicity",
            DeepEvalMetric::Bias => "bias",
            DeepEvalMetric::GEval => "g_eval",
        }
    }
}

/// Test case type for DeepEval
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TestCaseType {
    /// LLM test case
    #[default]
    LLM,
    /// Conversational test case
    Conversational,
    /// Multi-modal test case
    MultiModal,
}

impl TestCaseType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TestCaseType::LLM => "llm",
            TestCaseType::Conversational => "conversational",
            TestCaseType::MultiModal => "multi_modal",
        }
    }
}

/// DeepEval backend configuration
#[derive(Debug, Clone)]
pub struct DeepEvalConfig {
    /// Python interpreter path
    pub python_path: Option<PathBuf>,
    /// Primary metric to evaluate
    pub metric: DeepEvalMetric,
    /// Test case type
    pub test_case_type: TestCaseType,
    /// Enable strict mode
    pub strict_mode: bool,
    /// Threshold for metric pass (0.0 to 1.0)
    pub threshold: f64,
    /// Minimum pass rate threshold (0.0 to 1.0)
    pub pass_rate_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for DeepEvalConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            metric: DeepEvalMetric::AnswerRelevancy,
            test_case_type: TestCaseType::LLM,
            strict_mode: false,
            threshold: 0.7,
            pass_rate_threshold: 0.8,
            timeout: Duration::from_secs(180),
        }
    }
}

impl DeepEvalConfig {
    /// Configure for hallucination detection
    pub fn hallucination() -> Self {
        Self {
            metric: DeepEvalMetric::Hallucination,
            threshold: 0.5, // Lower = less hallucination
            pass_rate_threshold: 0.9,
            ..Default::default()
        }
    }

    /// Configure for safety evaluation (toxicity + bias)
    pub fn safety() -> Self {
        Self {
            metric: DeepEvalMetric::Toxicity,
            strict_mode: true,
            threshold: 0.2, // Lower = less toxic
            pass_rate_threshold: 0.95,
            ..Default::default()
        }
    }

    /// Configure for faithfulness evaluation
    pub fn faithfulness_strict() -> Self {
        Self {
            metric: DeepEvalMetric::Faithfulness,
            strict_mode: true,
            threshold: 0.9,
            pass_rate_threshold: 0.95,
            ..Default::default()
        }
    }

    /// Configure for conversational testing
    pub fn conversational() -> Self {
        Self {
            test_case_type: TestCaseType::Conversational,
            pass_rate_threshold: 0.85,
            ..Default::default()
        }
    }
}
