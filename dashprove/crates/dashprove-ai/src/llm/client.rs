//! LLM client trait and message types

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::LlmError;

/// Role of a message in a conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System message (instructions)
    System,
    /// User message
    User,
    /// Assistant response
    Assistant,
}

/// A message in an LLM conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    /// Role of the message sender
    pub role: MessageRole,
    /// Content of the message
    pub content: String,
}

impl LlmMessage {
    /// Create a new user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    /// Create a new assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }

    /// Create a new system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }
}

/// Response from an LLM completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// The generated text
    pub content: String,
    /// Model that generated the response
    pub model: String,
    /// Input tokens used
    pub input_tokens: Option<u32>,
    /// Output tokens generated
    pub output_tokens: Option<u32>,
    /// Stop reason (if available)
    pub stop_reason: Option<String>,
}

impl LlmResponse {
    /// Get total tokens used
    pub fn total_tokens(&self) -> Option<u32> {
        match (self.input_tokens, self.output_tokens) {
            (Some(i), Some(o)) => Some(i + o),
            _ => None,
        }
    }
}

/// Trait for LLM clients
///
/// This trait provides a unified interface for different LLM providers.
/// Implementations handle provider-specific API details.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Complete a single prompt
    async fn complete(&self, prompt: &str) -> Result<LlmResponse, LlmError>;

    /// Complete with conversation history
    async fn complete_messages(&self, messages: &[LlmMessage]) -> Result<LlmResponse, LlmError>;

    /// Check if the client is configured and ready
    fn is_configured(&self) -> bool;

    /// Get the model ID being used
    fn model_id(&self) -> &str;
}
