//! LLM client infrastructure for AI-powered proof assistance
//!
//! This module provides a unified interface for interacting with Large Language Models
//! (LLMs) to enable AI-powered features such as:
//!
//! - **Proof Synthesis**: Generate proof attempts from specifications
//! - **Tactic Prediction**: Suggest next proof steps with natural language reasoning
//! - **Spec Inference**: Infer specifications from code
//! - **Counterexample Explanation**: Natural language explanations of verification failures
//!
//! ## Supported Providers
//!
//! - **Anthropic Claude**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
//! - **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
//!
//! ## Usage
//!
//! ```rust,ignore
//! use dashprove_ai::llm::{LlmClient, AnthropicClient, LlmConfig};
//!
//! let config = LlmConfig::anthropic_default();
//! let client = AnthropicClient::new(config)?;
//!
//! let response = client.complete("Prove that addition is commutative").await?;
//! ```

mod client;
mod config;
mod provider;

pub use client::{LlmClient, LlmMessage, LlmResponse, MessageRole};
pub use config::{LlmConfig, LlmProvider, ModelId};
pub use provider::{create_client, try_create_default_client, AnthropicClient, OpenAiClient};

/// Error type for LLM operations
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// API request failed
    #[error("API request failed: {0}")]
    RequestFailed(String),

    /// Invalid API key or authentication failure
    #[error("Authentication failed: {0}")]
    AuthError(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded, retry after {retry_after_secs:?}s")]
    RateLimited { retry_after_secs: Option<u64> },

    /// Model not available or invalid model ID
    #[error("Invalid model: {0}")]
    InvalidModel(String),

    /// Response parsing error
    #[error("Failed to parse response: {0}")]
    ParseError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Context length exceeded
    #[error("Context length exceeded: {0} tokens")]
    ContextLengthExceeded(u32),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// LLM provider not configured
    #[error(
        "LLM provider not configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable"
    )]
    NotConfigured,
}

impl From<reqwest::Error> for LlmError {
    fn from(err: reqwest::Error) -> Self {
        LlmError::NetworkError(err.to_string())
    }
}
