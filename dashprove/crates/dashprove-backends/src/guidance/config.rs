//! Guidance backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Generation mode for structured output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenerationMode {
    /// Standard generation with constraints
    #[default]
    Constrained,
    /// Grammar-based generation (CFG)
    Grammar,
    /// Regex-based generation
    Regex,
    /// JSON schema generation
    JsonSchema,
}

impl GenerationMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            GenerationMode::Constrained => "constrained",
            GenerationMode::Grammar => "grammar",
            GenerationMode::Regex => "regex",
            GenerationMode::JsonSchema => "json_schema",
        }
    }
}

/// Validation strictness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationMode {
    /// Only check structure
    #[default]
    Structure,
    /// Check structure and types
    TypeChecked,
    /// Full semantic validation
    Semantic,
}

impl ValidationMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ValidationMode::Structure => "structure",
            ValidationMode::TypeChecked => "type_checked",
            ValidationMode::Semantic => "semantic",
        }
    }
}

/// Guidance backend configuration
#[derive(Debug, Clone)]
pub struct GuidanceConfig {
    /// Python interpreter path
    pub python_path: Option<PathBuf>,
    /// Generation mode
    pub generation_mode: GenerationMode,
    /// Validation mode
    pub validation_mode: ValidationMode,
    /// Allow partial matches
    pub allow_partial: bool,
    /// Maximum generation tokens
    pub max_tokens: u32,
    /// Pass rate threshold (0.0 to 1.0)
    pub pass_rate_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for GuidanceConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            generation_mode: GenerationMode::Constrained,
            validation_mode: ValidationMode::Structure,
            allow_partial: false,
            max_tokens: 512,
            pass_rate_threshold: 0.85,
            timeout: Duration::from_secs(120),
        }
    }
}

impl GuidanceConfig {
    /// Configure for JSON schema validation
    pub fn json_schema() -> Self {
        Self {
            generation_mode: GenerationMode::JsonSchema,
            validation_mode: ValidationMode::TypeChecked,
            pass_rate_threshold: 0.95,
            ..Default::default()
        }
    }

    /// Configure for grammar-based generation
    pub fn grammar() -> Self {
        Self {
            generation_mode: GenerationMode::Grammar,
            validation_mode: ValidationMode::Structure,
            allow_partial: true,
            pass_rate_threshold: 0.8,
            ..Default::default()
        }
    }

    /// Configure for strict semantic validation
    pub fn semantic_strict() -> Self {
        Self {
            validation_mode: ValidationMode::Semantic,
            allow_partial: false,
            pass_rate_threshold: 1.0,
            ..Default::default()
        }
    }
}
