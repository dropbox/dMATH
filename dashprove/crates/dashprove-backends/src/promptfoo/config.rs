//! Promptfoo backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Assertion type for prompt evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssertionType {
    /// Output contains expected substring
    #[default]
    Contains,
    /// Output equals expected value exactly
    Equals,
    /// Output matches regex pattern
    Regex,
    /// Output satisfies LLM-based evaluation
    LlmRubric,
    /// Output passes JSON schema validation
    JsonSchema,
    /// Output is similar to expected (semantic)
    Similar,
}

impl AssertionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            AssertionType::Contains => "contains",
            AssertionType::Equals => "equals",
            AssertionType::Regex => "regex",
            AssertionType::LlmRubric => "llm-rubric",
            AssertionType::JsonSchema => "is-json",
            AssertionType::Similar => "similar",
        }
    }
}

/// Output format for results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// JSON format
    #[default]
    Json,
    /// YAML format
    Yaml,
    /// CSV format
    Csv,
    /// HTML report
    Html,
}

impl OutputFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            OutputFormat::Json => "json",
            OutputFormat::Yaml => "yaml",
            OutputFormat::Csv => "csv",
            OutputFormat::Html => "html",
        }
    }
}

/// Promptfoo backend configuration
#[derive(Debug, Clone)]
pub struct PromptfooConfig {
    /// Node.js/npm executable path
    pub node_path: Option<PathBuf>,
    /// Default assertion type
    pub assertion_type: AssertionType,
    /// Output format for results
    pub output_format: OutputFormat,
    /// Number of test iterations per prompt
    pub iterations: u32,
    /// Maximum concurrent evaluations
    pub max_concurrency: u32,
    /// Pass rate threshold (0.0 to 1.0)
    pub pass_rate_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for PromptfooConfig {
    fn default() -> Self {
        Self {
            node_path: None,
            assertion_type: AssertionType::Contains,
            output_format: OutputFormat::Json,
            iterations: 1,
            max_concurrency: 4,
            pass_rate_threshold: 0.8,
            timeout: Duration::from_secs(300),
        }
    }
}

impl PromptfooConfig {
    /// Configure for strict equality testing
    pub fn strict() -> Self {
        Self {
            assertion_type: AssertionType::Equals,
            pass_rate_threshold: 1.0,
            ..Default::default()
        }
    }

    /// Configure for LLM-based evaluation
    pub fn llm_eval() -> Self {
        Self {
            assertion_type: AssertionType::LlmRubric,
            iterations: 3,
            pass_rate_threshold: 0.85,
            ..Default::default()
        }
    }

    /// Configure for JSON output validation
    pub fn json_validation() -> Self {
        Self {
            assertion_type: AssertionType::JsonSchema,
            pass_rate_threshold: 0.95,
            ..Default::default()
        }
    }
}
