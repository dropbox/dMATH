//! LangSmith backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Tracing mode for LangSmith observability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TracingMode {
    /// Full tracing of all LLM calls
    #[default]
    Full,
    /// Sample only (reduced overhead)
    Sample,
    /// Debug mode with verbose output
    Debug,
    /// Errors only
    ErrorsOnly,
}

impl TracingMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            TracingMode::Full => "full",
            TracingMode::Sample => "sample",
            TracingMode::Debug => "debug",
            TracingMode::ErrorsOnly => "errors_only",
        }
    }
}

/// Evaluation type for LangSmith
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvaluationType {
    /// Custom evaluator
    #[default]
    Custom,
    /// LLM-as-judge evaluation
    LLMJudge,
    /// Embedding similarity
    Similarity,
    /// Exact match
    ExactMatch,
    /// Regex match
    Regex,
}

impl EvaluationType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EvaluationType::Custom => "custom",
            EvaluationType::LLMJudge => "llm_judge",
            EvaluationType::Similarity => "similarity",
            EvaluationType::ExactMatch => "exact_match",
            EvaluationType::Regex => "regex",
        }
    }
}

/// LangSmith backend configuration
#[derive(Debug, Clone)]
pub struct LangSmithConfig {
    /// Python interpreter path
    pub python_path: Option<PathBuf>,
    /// Tracing mode
    pub tracing_mode: TracingMode,
    /// Evaluation type
    pub evaluation_type: EvaluationType,
    /// Enable feedback collection
    pub enable_feedback: bool,
    /// Enable dataset comparisons
    pub enable_comparisons: bool,
    /// Sample rate (0.0 to 1.0)
    pub sample_rate: f64,
    /// Minimum pass rate threshold (0.0 to 1.0)
    pub pass_rate_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for LangSmithConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            tracing_mode: TracingMode::Full,
            evaluation_type: EvaluationType::Custom,
            enable_feedback: true,
            enable_comparisons: false,
            sample_rate: 1.0,
            pass_rate_threshold: 0.8,
            timeout: Duration::from_secs(120),
        }
    }
}

impl LangSmithConfig {
    /// Configure for LLM-as-judge evaluation
    pub fn llm_judge() -> Self {
        Self {
            evaluation_type: EvaluationType::LLMJudge,
            pass_rate_threshold: 0.85,
            ..Default::default()
        }
    }

    /// Configure for similarity-based evaluation
    pub fn similarity() -> Self {
        Self {
            evaluation_type: EvaluationType::Similarity,
            pass_rate_threshold: 0.9,
            ..Default::default()
        }
    }

    /// Configure for sampled tracing
    pub fn sampled(rate: f64) -> Self {
        Self {
            tracing_mode: TracingMode::Sample,
            sample_rate: rate.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Configure for strict exact matching
    pub fn exact_match_strict() -> Self {
        Self {
            evaluation_type: EvaluationType::ExactMatch,
            pass_rate_threshold: 1.0,
            ..Default::default()
        }
    }
}
