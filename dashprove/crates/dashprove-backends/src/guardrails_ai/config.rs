//! GuardrailsAI backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Guardrail type for LLM output validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GuardrailType {
    /// Validate format/schema compliance
    #[default]
    Schema,
    /// Content quality validation
    Quality,
    /// Safety/toxicity filtering
    Safety,
    /// Factual accuracy checking
    Factual,
    /// Custom validator
    Custom,
}

impl GuardrailType {
    pub fn as_str(&self) -> &'static str {
        match self {
            GuardrailType::Schema => "schema",
            GuardrailType::Quality => "quality",
            GuardrailType::Safety => "safety",
            GuardrailType::Factual => "factual",
            GuardrailType::Custom => "custom",
        }
    }
}

/// Validation strictness level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrictnessLevel {
    /// Lenient validation - allow minor issues
    Lenient,
    /// Standard validation
    #[default]
    Standard,
    /// Strict validation - reject any issues
    Strict,
}

impl StrictnessLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            StrictnessLevel::Lenient => "lenient",
            StrictnessLevel::Standard => "standard",
            StrictnessLevel::Strict => "strict",
        }
    }
}

/// GuardrailsAI backend configuration
#[derive(Debug, Clone)]
pub struct GuardrailsAIConfig {
    /// Python interpreter path
    pub python_path: Option<PathBuf>,
    /// Type of guardrail to apply
    pub guardrail_type: GuardrailType,
    /// Validation strictness level
    pub strictness: StrictnessLevel,
    /// Whether to use on-fail fallbacks
    pub use_fallbacks: bool,
    /// Maximum retries for validation
    pub max_retries: u32,
    /// Minimum pass rate threshold (0.0 to 1.0)
    pub pass_rate_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for GuardrailsAIConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            guardrail_type: GuardrailType::Schema,
            strictness: StrictnessLevel::Standard,
            use_fallbacks: true,
            max_retries: 3,
            pass_rate_threshold: 0.8,
            timeout: Duration::from_secs(120),
        }
    }
}

impl GuardrailsAIConfig {
    /// Configure for safety validation
    pub fn safety() -> Self {
        Self {
            guardrail_type: GuardrailType::Safety,
            strictness: StrictnessLevel::Strict,
            pass_rate_threshold: 0.95,
            ..Default::default()
        }
    }

    /// Configure for schema validation
    pub fn schema_strict() -> Self {
        Self {
            guardrail_type: GuardrailType::Schema,
            strictness: StrictnessLevel::Strict,
            use_fallbacks: false,
            pass_rate_threshold: 1.0,
            ..Default::default()
        }
    }
}
