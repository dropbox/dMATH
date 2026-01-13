//! LLM provider implementations

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::{
    client::{LlmClient, LlmMessage, LlmResponse, MessageRole},
    config::LlmConfig,
    LlmError,
};

/// Anthropic Claude API client
pub struct AnthropicClient {
    client: reqwest::Client,
    config: LlmConfig,
    api_key: String,
}

impl AnthropicClient {
    /// Create a new Anthropic client
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        let api_key = config.get_api_key().ok_or(LlmError::ConfigError(
            "ANTHROPIC_API_KEY not set".to_string(),
        ))?;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| LlmError::ConfigError(e.to_string()))?;

        Ok(Self {
            client,
            config,
            api_key,
        })
    }

    /// Try to create a client, returning None if not configured
    pub fn try_new(config: LlmConfig) -> Option<Self> {
        Self::new(config).ok()
    }
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    temperature: f32,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    model: String,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Deserialize)]
struct AnthropicError {
    error: AnthropicErrorDetail,
}

#[derive(Deserialize)]
struct AnthropicErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn complete(&self, prompt: &str) -> Result<LlmResponse, LlmError> {
        self.complete_messages(&[LlmMessage::user(prompt)]).await
    }

    async fn complete_messages(&self, messages: &[LlmMessage]) -> Result<LlmResponse, LlmError> {
        // Convert messages, handling system separately
        let mut system_prompt = self.config.system_prompt.clone();
        let mut api_messages = Vec::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    system_prompt = Some(msg.content.clone());
                }
                MessageRole::User => {
                    api_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: msg.content.clone(),
                    });
                }
                MessageRole::Assistant => {
                    api_messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: msg.content.clone(),
                    });
                }
            }
        }

        let request = AnthropicRequest {
            model: self.config.model.api_name().to_string(),
            max_tokens: self.config.max_tokens,
            system: system_prompt,
            messages: api_messages,
            temperature: self.config.temperature,
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            // Try to parse error response
            if let Ok(error) = serde_json::from_str::<AnthropicError>(&body) {
                return match error.error.error_type.as_str() {
                    "authentication_error" => Err(LlmError::AuthError(error.error.message)),
                    "rate_limit_error" => Err(LlmError::RateLimited {
                        retry_after_secs: None,
                    }),
                    "invalid_request_error" if error.error.message.contains("context length") => {
                        Err(LlmError::ContextLengthExceeded(0))
                    }
                    _ => Err(LlmError::RequestFailed(error.error.message)),
                };
            }
            return Err(LlmError::RequestFailed(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        let parsed: AnthropicResponse =
            serde_json::from_str(&body).map_err(|e| LlmError::ParseError(e.to_string()))?;

        let content = parsed
            .content
            .into_iter()
            .filter_map(|c| {
                if c.content_type == "text" {
                    c.text
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(LlmResponse {
            content,
            model: parsed.model,
            input_tokens: Some(parsed.usage.input_tokens),
            output_tokens: Some(parsed.usage.output_tokens),
            stop_reason: parsed.stop_reason,
        })
    }

    fn is_configured(&self) -> bool {
        true // If we got here, we have an API key
    }

    fn model_id(&self) -> &str {
        self.config.model.api_name()
    }
}

/// OpenAI GPT API client
pub struct OpenAiClient {
    client: reqwest::Client,
    config: LlmConfig,
    api_key: String,
}

impl OpenAiClient {
    /// Create a new OpenAI client
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        let api_key = config
            .get_api_key()
            .ok_or(LlmError::ConfigError("OPENAI_API_KEY not set".to_string()))?;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| LlmError::ConfigError(e.to_string()))?;

        Ok(Self {
            client,
            config,
            api_key,
        })
    }

    /// Try to create a client, returning None if not configured
    pub fn try_new(config: LlmConfig) -> Option<Self> {
        Self::new(config).ok()
    }
}

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    model: String,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiResponseMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Deserialize)]
struct OpenAiError {
    error: OpenAiErrorDetail,
}

#[derive(Deserialize)]
struct OpenAiErrorDetail {
    message: String,
    #[allow(dead_code)] // Used for debugging/logging
    #[serde(rename = "type")]
    error_type: Option<String>,
    code: Option<String>,
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(&self, prompt: &str) -> Result<LlmResponse, LlmError> {
        self.complete_messages(&[LlmMessage::user(prompt)]).await
    }

    async fn complete_messages(&self, messages: &[LlmMessage]) -> Result<LlmResponse, LlmError> {
        let mut api_messages = Vec::new();

        // Add system prompt if configured
        if let Some(ref system) = self.config.system_prompt {
            api_messages.push(OpenAiMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        // Convert messages
        for msg in messages {
            let role = match msg.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
            };
            api_messages.push(OpenAiMessage {
                role: role.to_string(),
                content: msg.content.clone(),
            });
        }

        let request = OpenAiRequest {
            model: self.config.model.api_name().to_string(),
            messages: api_messages,
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            if let Ok(error) = serde_json::from_str::<OpenAiError>(&body) {
                return match error.error.code.as_deref() {
                    Some("invalid_api_key") => Err(LlmError::AuthError(error.error.message)),
                    Some("rate_limit_exceeded") => Err(LlmError::RateLimited {
                        retry_after_secs: None,
                    }),
                    Some("context_length_exceeded") => Err(LlmError::ContextLengthExceeded(0)),
                    _ => Err(LlmError::RequestFailed(error.error.message)),
                };
            }
            return Err(LlmError::RequestFailed(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        let parsed: OpenAiResponse =
            serde_json::from_str(&body).map_err(|e| LlmError::ParseError(e.to_string()))?;

        let content = parsed
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let stop_reason = parsed.choices.first().and_then(|c| c.finish_reason.clone());

        Ok(LlmResponse {
            content,
            model: parsed.model,
            input_tokens: parsed.usage.as_ref().map(|u| u.prompt_tokens),
            output_tokens: parsed.usage.as_ref().map(|u| u.completion_tokens),
            stop_reason,
        })
    }

    fn is_configured(&self) -> bool {
        true
    }

    fn model_id(&self) -> &str {
        self.config.model.api_name()
    }
}

/// Create an LLM client from config, automatically selecting provider
pub fn create_client(config: LlmConfig) -> Result<Box<dyn LlmClient>, LlmError> {
    match config.provider() {
        super::config::LlmProvider::Anthropic => Ok(Box::new(AnthropicClient::new(config)?)),
        super::config::LlmProvider::OpenAi => Ok(Box::new(OpenAiClient::new(config)?)),
    }
}

/// Try to create a client from environment, preferring Anthropic
pub fn try_create_default_client() -> Option<Box<dyn LlmClient>> {
    // Try Anthropic first
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        if let Ok(client) = AnthropicClient::new(LlmConfig::anthropic_default()) {
            return Some(Box::new(client));
        }
    }

    // Fall back to OpenAI
    if std::env::var("OPENAI_API_KEY").is_ok() {
        if let Ok(client) = OpenAiClient::new(LlmConfig::openai_default()) {
            return Some(Box::new(client));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let user = LlmMessage::user("Hello");
        assert_eq!(user.role, MessageRole::User);
        assert_eq!(user.content, "Hello");

        let assistant = LlmMessage::assistant("Hi there");
        assert_eq!(assistant.role, MessageRole::Assistant);
    }

    #[test]
    fn test_config_defaults() {
        let config = LlmConfig::default();
        assert_eq!(config.max_tokens, 4096);
        assert!(config.temperature < 0.2); // Low for determinism
    }

    #[test]
    fn test_model_api_names() {
        use super::super::config::ModelId;

        assert_eq!(
            ModelId::Claude35Sonnet.api_name(),
            "claude-3-5-sonnet-20241022"
        );
        assert_eq!(ModelId::Gpt4Turbo.api_name(), "gpt-4-turbo");
    }

    #[test]
    fn test_try_create_without_key() {
        // Should return None when no API keys are set
        // (assuming test environment doesn't have them)
        let _ = try_create_default_client(); // Just verify it doesn't panic
    }
}
