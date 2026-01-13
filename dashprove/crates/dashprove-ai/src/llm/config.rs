//! LLM configuration types

use serde::{Deserialize, Serialize};

/// LLM provider selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LlmProvider {
    /// Anthropic Claude models
    #[default]
    Anthropic,
    /// OpenAI GPT models
    OpenAi,
}

/// Model identifier for each provider
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelId {
    // Anthropic models
    /// Claude 3.5 Sonnet (default for Anthropic)
    #[default]
    Claude35Sonnet,
    /// Claude 3 Opus (highest capability)
    Claude3Opus,
    /// Claude 3 Haiku (fastest)
    Claude3Haiku,
    /// Claude 3.5 Haiku
    Claude35Haiku,

    // OpenAI models
    /// GPT-4 Turbo (default for OpenAI)
    Gpt4Turbo,
    /// GPT-4
    Gpt4,
    /// GPT-4o
    Gpt4o,
    /// GPT-4o mini
    Gpt4oMini,
    /// GPT-3.5 Turbo
    Gpt35Turbo,

    /// Custom model ID string
    Custom(String),
}

impl ModelId {
    /// Get the API model string for this model
    pub fn api_name(&self) -> &str {
        match self {
            ModelId::Claude35Sonnet => "claude-3-5-sonnet-20241022",
            ModelId::Claude3Opus => "claude-3-opus-20240229",
            ModelId::Claude3Haiku => "claude-3-haiku-20240307",
            ModelId::Claude35Haiku => "claude-3-5-haiku-20241022",
            ModelId::Gpt4Turbo => "gpt-4-turbo",
            ModelId::Gpt4 => "gpt-4",
            ModelId::Gpt4o => "gpt-4o",
            ModelId::Gpt4oMini => "gpt-4o-mini",
            ModelId::Gpt35Turbo => "gpt-3.5-turbo",
            ModelId::Custom(s) => s,
        }
    }

    /// Get the provider for this model
    pub fn provider(&self) -> LlmProvider {
        match self {
            ModelId::Claude35Sonnet
            | ModelId::Claude3Opus
            | ModelId::Claude3Haiku
            | ModelId::Claude35Haiku => LlmProvider::Anthropic,
            ModelId::Gpt4Turbo
            | ModelId::Gpt4
            | ModelId::Gpt4o
            | ModelId::Gpt4oMini
            | ModelId::Gpt35Turbo => LlmProvider::OpenAi,
            ModelId::Custom(_) => LlmProvider::OpenAi, // Default to OpenAI for custom
        }
    }

    /// Get the maximum context length for this model
    pub fn max_context_tokens(&self) -> u32 {
        match self {
            ModelId::Claude35Sonnet => 200_000,
            ModelId::Claude3Opus => 200_000,
            ModelId::Claude3Haiku => 200_000,
            ModelId::Claude35Haiku => 200_000,
            ModelId::Gpt4Turbo => 128_000,
            ModelId::Gpt4 => 8_192,
            ModelId::Gpt4o => 128_000,
            ModelId::Gpt4oMini => 128_000,
            ModelId::Gpt35Turbo => 16_385,
            ModelId::Custom(_) => 8_192, // Conservative default
        }
    }
}

/// LLM client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// API key (or will be read from environment)
    pub api_key: Option<String>,
    /// Model to use
    pub model: ModelId,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature (0.0 - 1.0)
    pub temperature: f32,
    /// System prompt to use
    pub system_prompt: Option<String>,
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            model: ModelId::default(),
            max_tokens: 4096,
            temperature: 0.1, // Low temperature for deterministic proof generation
            system_prompt: None,
            timeout_secs: 120,
        }
    }
}

impl LlmConfig {
    /// Create a new config with the given model
    pub fn with_model(model: ModelId) -> Self {
        Self {
            model,
            ..Default::default()
        }
    }

    /// Create default Anthropic configuration
    pub fn anthropic_default() -> Self {
        Self {
            model: ModelId::Claude35Sonnet,
            ..Default::default()
        }
    }

    /// Create default OpenAI configuration
    pub fn openai_default() -> Self {
        Self {
            model: ModelId::Gpt4Turbo,
            ..Default::default()
        }
    }

    /// Set the API key
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.clamp(0.0, 1.0);
        self
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Get the provider for this config
    pub fn provider(&self) -> LlmProvider {
        self.model.provider()
    }

    /// Get API key from config or environment
    pub fn get_api_key(&self) -> Option<String> {
        self.api_key.clone().or_else(|| match self.provider() {
            LlmProvider::Anthropic => std::env::var("ANTHROPIC_API_KEY").ok(),
            LlmProvider::OpenAi => std::env::var("OPENAI_API_KEY").ok(),
        })
    }
}
