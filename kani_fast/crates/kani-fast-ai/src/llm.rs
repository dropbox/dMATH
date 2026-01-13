//! LLM Integration for invariant suggestions
//!
//! This module provides integration with language models for suggesting
//! loop invariants when template-based and ICE learning methods fail.

use kani_fast_kinduction::{Property, StateFormula, TransitionSystem};
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt::Write;

use crate::ice::Example;
use crate::AiError;

// Pre-compiled regexes for parsing LLM responses
lazy_static! {
    /// Matches SMT code blocks: ```smt ... ```
    static ref RE_SMT_BLOCK: Regex = Regex::new(r"```smt\s*\n([^`]+)\n```").unwrap();
    /// Matches S-expression invariants (fallback without code blocks)
    static ref RE_SEXP: Regex = Regex::new(r"\((?:>=|<=|>|<|=|and|or|not|=>)[^)]+\)").unwrap();
}

/// LLM provider selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmProvider {
    /// OpenAI API (GPT-4, etc.)
    OpenAI,
    /// Anthropic API (Claude)
    Anthropic,
    /// Local model via Ollama
    Ollama,
    /// Mock provider for testing
    Mock,
}

impl Default for LlmProvider {
    fn default() -> Self {
        // Auto-detect based on available API keys
        if env::var("OPENAI_API_KEY").is_ok() {
            Self::OpenAI
        } else if env::var("ANTHROPIC_API_KEY").is_ok() {
            Self::Anthropic
        } else {
            Self::Ollama
        }
    }
}

/// Configuration for LLM integration
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// LLM provider to use
    pub provider: LlmProvider,
    /// Model name (e.g., "gpt-4", "claude-3-opus")
    pub model: String,
    /// API base URL (for custom endpoints)
    pub api_base: Option<String>,
    /// Maximum tokens in response
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Number of invariant suggestions per query
    pub num_suggestions: usize,
}

impl Default for LlmConfig {
    fn default() -> Self {
        let provider = LlmProvider::default();
        let model = match provider {
            LlmProvider::OpenAI => "gpt-4".to_string(),
            LlmProvider::Anthropic => "claude-3-opus-20240229".to_string(),
            LlmProvider::Ollama => "codellama".to_string(),
            LlmProvider::Mock => "mock".to_string(),
        };

        Self {
            provider,
            model,
            api_base: None,
            max_tokens: 1024,
            temperature: 0.2,
            num_suggestions: 3,
        }
    }
}

/// LLM client for invariant synthesis
pub struct LlmClient {
    config: LlmConfig,
    client: reqwest::Client,
}

impl LlmClient {
    /// Create a new LLM client
    pub fn new(config: LlmConfig) -> Self {
        let client = reqwest::Client::new();
        Self { config, client }
    }

    /// Suggest invariants for the given system and property
    pub async fn suggest_invariants(
        &self,
        system: &TransitionSystem,
        property: &Property,
        examples: &[Example],
    ) -> Result<Vec<StateFormula>, AiError> {
        let prompt = self.build_prompt(system, property, examples);

        let response = match self.config.provider {
            LlmProvider::OpenAI => self.query_openai(&prompt).await?,
            LlmProvider::Anthropic => self.query_anthropic(&prompt).await?,
            LlmProvider::Ollama => self.query_ollama(&prompt).await?,
            LlmProvider::Mock => self.mock_response(system),
        };

        self.parse_invariants(&response)
    }

    /// Build the prompt for invariant synthesis
    fn build_prompt(
        &self,
        system: &TransitionSystem,
        property: &Property,
        examples: &[Example],
    ) -> String {
        let mut prompt = String::new();

        prompt
            .push_str("You are an expert in formal verification and loop invariant synthesis.\n\n");
        prompt.push_str("Given the following transition system, suggest loop invariants that:\n");
        prompt.push_str("1. Are satisfied by the initial state\n");
        prompt.push_str("2. Are preserved by the transition relation\n");
        prompt.push_str("3. Imply the property to be verified\n\n");

        prompt.push_str("## Transition System\n\n");
        prompt.push_str("### Variables\n");
        for var in &system.variables {
            let _ = writeln!(prompt, "- {} : {}", var.name, var.smt_type.to_smt_string());
        }

        prompt.push_str("\n### Initial State\n");
        let _ = writeln!(prompt, "```\n{}\n```\n", system.init.smt_formula);

        prompt.push_str("### Transition Relation\n");
        let _ = writeln!(prompt, "```\n{}\n```\n", system.transition.smt_formula);

        prompt.push_str("### Property to Verify\n");
        let _ = writeln!(prompt, "```\n{}\n```\n", property.formula.smt_formula);

        if !examples.is_empty() {
            prompt.push_str("## Examples from ICE Learning\n\n");
            prompt
                .push_str("The following examples show states that the invariant must handle:\n\n");

            for (i, example) in examples.iter().enumerate() {
                let kind_str = match &example.kind {
                    crate::ice::ExampleKind::Positive => "POSITIVE (must satisfy invariant)",
                    crate::ice::ExampleKind::Negative => "NEGATIVE (must NOT satisfy invariant)",
                    crate::ice::ExampleKind::Implication { .. } => {
                        "IMPLICATION (if pre satisfies, post must satisfy)"
                    }
                };

                let _ = writeln!(prompt, "Example {}: {}", i + 1, kind_str);
                let _ = writeln!(prompt, "  Values: {:?}", example.values);

                if let crate::ice::ExampleKind::Implication { post } = &example.kind {
                    let _ = writeln!(prompt, "  Post-values: {post:?}");
                }
                prompt.push('\n');
            }
        }

        prompt.push_str("## Output Format\n\n");
        let _ = writeln!(
            prompt,
            "Provide exactly {} invariant suggestions in SMT-LIB2 format.",
            self.config.num_suggestions
        );
        prompt.push_str("Each invariant should be on its own line, wrapped in ```smt markers.\n");
        prompt.push_str("Example output:\n");
        prompt.push_str("```smt\n(>= x 0)\n```\n");
        prompt.push_str("```smt\n(and (>= x 0) (<= x n))\n```\n");

        prompt
    }

    /// Query OpenAI API
    async fn query_openai(&self, prompt: &str) -> Result<String, AiError> {
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| AiError::LlmApi("OPENAI_API_KEY not set".to_string()))?;

        let api_base = self
            .config
            .api_base
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in formal verification and SMT-LIB2 syntax."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        });

        let response = self
            .client
            .post(format!("{api_base}/chat/completions"))
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AiError::LlmApi(format!("Request failed: {e}")))?;

        let response_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| AiError::LlmApi(format!("Failed to parse response: {e}")))?;

        response_json["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| AiError::LlmApi("No content in response".to_string()))
    }

    /// Query Anthropic API
    async fn query_anthropic(&self, prompt: &str) -> Result<String, AiError> {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .map_err(|_| AiError::LlmApi("ANTHROPIC_API_KEY not set".to_string()))?;

        let api_base = self
            .config
            .api_base
            .clone()
            .unwrap_or_else(|| "https://api.anthropic.com/v1".to_string());

        let body = serde_json::json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        });

        let response = self
            .client
            .post(format!("{api_base}/messages"))
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AiError::LlmApi(format!("Request failed: {e}")))?;

        let response_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| AiError::LlmApi(format!("Failed to parse response: {e}")))?;

        response_json["content"][0]["text"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| AiError::LlmApi("No content in response".to_string()))
    }

    /// Query local Ollama
    async fn query_ollama(&self, prompt: &str) -> Result<String, AiError> {
        let api_base = self
            .config
            .api_base
            .clone()
            .unwrap_or_else(|| "http://localhost:11434".to_string());

        let body = serde_json::json!({
            "model": self.config.model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        });

        let response = self
            .client
            .post(format!("{api_base}/api/generate"))
            .json(&body)
            .send()
            .await
            .map_err(|e| AiError::LlmApi(format!("Ollama request failed: {e}")))?;

        let response_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| AiError::LlmApi(format!("Failed to parse response: {e}")))?;

        response_json["response"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| AiError::LlmApi("No response from Ollama".to_string()))
    }

    /// Mock response for testing
    fn mock_response(&self, system: &TransitionSystem) -> String {
        let mut result = String::new();

        // Generate simple invariants based on variables
        for var in &system.variables {
            if var.smt_type == kani_fast_kinduction::SmtType::Int {
                let _ = writeln!(result, "```smt\n(>= {} 0)\n```", var.name);
            }
        }

        result
    }

    /// Parse invariants from LLM response
    fn parse_invariants(&self, response: &str) -> Result<Vec<StateFormula>, AiError> {
        let mut invariants = Vec::new();

        // Look for ```smt ... ``` blocks using pre-compiled regex
        for cap in RE_SMT_BLOCK.captures_iter(response) {
            if let Some(formula_match) = cap.get(1) {
                let formula = formula_match.as_str().trim();
                if !formula.is_empty() && self.is_valid_smt(formula) {
                    invariants.push(StateFormula::with_description(
                        formula.to_string(),
                        "LLM suggestion".to_string(),
                    ));
                }
            }
        }

        // Also try to find formulas without code blocks (fallback)
        if invariants.is_empty() {
            // Look for S-expressions that look like invariants using pre-compiled regex
            for mat in RE_SEXP.find_iter(response) {
                let formula = mat.as_str();
                if self.is_valid_smt(formula) {
                    invariants.push(StateFormula::with_description(
                        formula.to_string(),
                        "LLM suggestion".to_string(),
                    ));
                }
            }
        }

        Ok(invariants)
    }

    /// Basic validation of SMT formula
    fn is_valid_smt(&self, formula: &str) -> bool {
        // Check balanced parentheses
        let mut depth = 0;
        for c in formula.chars() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        return false;
                    }
                }
                _ => {}
            }
        }
        depth == 0 && formula.starts_with('(')
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_config_default() {
        let config = LlmConfig::default();
        assert_eq!(config.num_suggestions, 3);
        assert!(config.temperature > 0.0);
    }

    #[test]
    fn test_mock_provider() {
        use kani_fast_kinduction::{SmtType, TransitionSystemBuilder};

        let config = LlmConfig {
            provider: LlmProvider::Mock,
            ..Default::default()
        };
        let client = LlmClient::new(config);

        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();

        let response = client.mock_response(&system);
        assert!(response.contains("(>= x 0)"));
    }

    #[test]
    fn test_parse_invariants() {
        let config = LlmConfig {
            provider: LlmProvider::Mock,
            ..Default::default()
        };
        let client = LlmClient::new(config);

        let response = r"
Here are some invariant suggestions:

```smt
(>= x 0)
```

```smt
(and (>= x 0) (<= x n))
```

The first invariant is simpler...
";

        let invariants = client.parse_invariants(response).unwrap();
        assert_eq!(invariants.len(), 2);
        assert_eq!(invariants[0].smt_formula, "(>= x 0)");
    }

    #[test]
    fn test_is_valid_smt() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        assert!(client.is_valid_smt("(>= x 0)"));
        assert!(client.is_valid_smt("(and (>= x 0) (<= x 10))"));
        assert!(!client.is_valid_smt("x >= 0")); // Not S-expression
        assert!(!client.is_valid_smt("(>= x 0")); // Unbalanced
        assert!(!client.is_valid_smt("(>= x 0))")); // Unbalanced
    }

    #[test]
    fn test_build_prompt() {
        use kani_fast_kinduction::{SmtType, TransitionSystemBuilder};

        let config = LlmConfig {
            provider: LlmProvider::Mock,
            num_suggestions: 2,
            ..Default::default()
        };
        let client = LlmClient::new(config);

        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();

        let property = Property::safety("p1", "non_negative", StateFormula::new("(>= x 0)"));

        let prompt = client.build_prompt(&system, &property, &[]);

        assert!(prompt.contains("Transition System"));
        assert!(prompt.contains("x : Int"));
        assert!(prompt.contains("(= x 0)"));
        assert!(prompt.contains("2 invariant suggestions"));
    }

    #[test]
    fn test_llm_provider_default_no_keys() {
        // When no API keys are set, default should be Ollama
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("ANTHROPIC_API_KEY");
        let provider = LlmProvider::default();
        assert_eq!(provider, LlmProvider::Ollama);
    }

    #[test]
    fn test_llm_provider_equality() {
        assert_eq!(LlmProvider::OpenAI, LlmProvider::OpenAI);
        assert_eq!(LlmProvider::Anthropic, LlmProvider::Anthropic);
        assert_eq!(LlmProvider::Ollama, LlmProvider::Ollama);
        assert_eq!(LlmProvider::Mock, LlmProvider::Mock);
        assert_ne!(LlmProvider::OpenAI, LlmProvider::Anthropic);
    }

    #[test]
    fn test_llm_provider_debug() {
        let provider = LlmProvider::Mock;
        let debug_str = format!("{:?}", provider);
        assert!(debug_str.contains("Mock"));
    }

    #[test]
    fn test_llm_provider_clone() {
        let provider = LlmProvider::OpenAI;
        let cloned = provider;
        assert_eq!(cloned, LlmProvider::OpenAI);
    }

    #[test]
    fn test_llm_provider_copy() {
        let provider = LlmProvider::Anthropic;
        let copied: LlmProvider = provider;
        assert_eq!(copied, LlmProvider::Anthropic);
    }

    #[test]
    fn test_llm_config_custom() {
        let config = LlmConfig {
            provider: LlmProvider::OpenAI,
            model: "gpt-4-turbo".to_string(),
            api_base: Some("https://custom.api.com".to_string()),
            max_tokens: 2048,
            temperature: 0.5,
            num_suggestions: 5,
        };
        assert_eq!(config.provider, LlmProvider::OpenAI);
        assert_eq!(config.model, "gpt-4-turbo");
        assert_eq!(config.api_base, Some("https://custom.api.com".to_string()));
        assert_eq!(config.max_tokens, 2048);
        assert!((config.temperature - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.num_suggestions, 5);
    }

    #[test]
    fn test_llm_config_clone() {
        let config = LlmConfig {
            provider: LlmProvider::Mock,
            model: "test-model".to_string(),
            api_base: None,
            max_tokens: 512,
            temperature: 0.1,
            num_suggestions: 2,
        };
        let cloned = config.clone();
        assert_eq!(cloned.provider, LlmProvider::Mock);
        assert_eq!(cloned.model, "test-model");
        assert!(cloned.api_base.is_none());
        assert_eq!(cloned.max_tokens, 512);
        assert_eq!(cloned.num_suggestions, 2);
    }

    #[test]
    fn test_llm_config_debug() {
        let config = LlmConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("LlmConfig"));
        assert!(debug_str.contains("num_suggestions"));
    }

    #[test]
    fn test_llm_client_creation() {
        let config = LlmConfig {
            provider: LlmProvider::Mock,
            ..Default::default()
        };
        let _client = LlmClient::new(config);
        // Client should be created successfully
    }

    #[test]
    fn test_mock_response_multiple_vars() {
        use kani_fast_kinduction::{SmtType, TransitionSystemBuilder};

        let config = LlmConfig {
            provider: LlmProvider::Mock,
            ..Default::default()
        };
        let client = LlmClient::new(config);

        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .variable("y", SmtType::Int)
            .variable("z", SmtType::Int)
            .init("(and (= x 0) (= y 0) (= z 0))")
            .transition("true")
            .build();

        let response = client.mock_response(&system);
        assert!(response.contains("(>= x 0)"));
        assert!(response.contains("(>= y 0)"));
        assert!(response.contains("(>= z 0)"));
    }

    #[test]
    fn test_mock_response_non_int_vars() {
        use kani_fast_kinduction::{SmtType, TransitionSystemBuilder};

        let config = LlmConfig {
            provider: LlmProvider::Mock,
            ..Default::default()
        };
        let client = LlmClient::new(config);

        let system = TransitionSystemBuilder::new()
            .variable("b", SmtType::Bool)
            .variable("x", SmtType::Int)
            .init("true")
            .transition("true")
            .build();

        let response = client.mock_response(&system);
        // Should only generate for Int variables
        assert!(response.contains("(>= x 0)"));
        assert!(!response.contains("(>= b 0)"));
    }

    #[test]
    fn test_parse_invariants_empty_response() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        let response = "No invariants found.";
        let invariants = client.parse_invariants(response).unwrap();
        assert!(invariants.is_empty());
    }

    #[test]
    fn test_parse_invariants_fallback_sexp() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        // Without code blocks, should try to parse S-expressions
        let response = "The invariant is (>= x 0) and also (and (>= y 0) (<= y 10))";
        let invariants = client.parse_invariants(response).unwrap();
        assert!(!invariants.is_empty());
    }

    #[test]
    fn test_parse_invariants_single() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        let response = "```smt\n(<= x 100)\n```";
        let invariants = client.parse_invariants(response).unwrap();
        assert_eq!(invariants.len(), 1);
        assert_eq!(invariants[0].smt_formula, "(<= x 100)");
    }

    #[test]
    fn test_parse_invariants_with_description() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        let response = "```smt\n(>= x 0)\n```";
        let invariants = client.parse_invariants(response).unwrap();
        assert_eq!(invariants.len(), 1);
        assert_eq!(
            invariants[0].description,
            Some("LLM suggestion".to_string())
        );
    }

    #[test]
    fn test_is_valid_smt_nested() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        assert!(client.is_valid_smt("(and (>= x 0) (or (<= y 10) (= z 5)))"));
        assert!(client.is_valid_smt("(not (and (= x 0) (= y 0)))"));
    }

    #[test]
    fn test_is_valid_smt_deep_nesting() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        assert!(client.is_valid_smt("(a (b (c (d (e)))))"));
    }

    #[test]
    fn test_is_valid_smt_empty() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        assert!(!client.is_valid_smt(""));
        assert!(!client.is_valid_smt("   "));
    }

    #[test]
    fn test_is_valid_smt_only_parens() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        assert!(client.is_valid_smt("()"));
        assert!(client.is_valid_smt("(())"));
    }

    #[test]
    fn test_is_valid_smt_mismatched_parens() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        assert!(!client.is_valid_smt("(()")); // Too many open
        assert!(!client.is_valid_smt("())")); // Too many close
        assert!(!client.is_valid_smt(")(")); // Wrong order
    }

    #[test]
    fn test_build_prompt_with_examples() {
        use crate::ice::Example;
        use kani_fast_kinduction::{SmtType, TransitionSystemBuilder};
        use std::collections::HashMap;

        let config = LlmConfig {
            provider: LlmProvider::Mock,
            num_suggestions: 3,
            ..Default::default()
        };
        let client = LlmClient::new(config);

        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();

        let property = Property::safety("p1", "non_negative", StateFormula::new("(>= x 0)"));

        let examples = vec![
            Example::positive(HashMap::from([("x".to_string(), 5)])),
            Example::negative(HashMap::from([("x".to_string(), -1)])),
        ];

        let prompt = client.build_prompt(&system, &property, &examples);

        assert!(prompt.contains("Examples from ICE Learning"));
        assert!(prompt.contains("POSITIVE"));
        assert!(prompt.contains("NEGATIVE"));
    }

    #[test]
    fn test_build_prompt_with_implication_example() {
        use crate::ice::Example;
        use kani_fast_kinduction::{SmtType, TransitionSystemBuilder};
        use std::collections::HashMap;

        let config = LlmConfig {
            provider: LlmProvider::Mock,
            ..Default::default()
        };
        let client = LlmClient::new(config);

        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("(= x' (+ x 1))")
            .build();

        let property = Property::safety("p1", "non_negative", StateFormula::new("(>= x 0)"));

        let pre = HashMap::from([("x".to_string(), 5)]);
        let post = HashMap::from([("x".to_string(), 6)]);
        let examples = vec![Example::implication(pre, post)];

        let prompt = client.build_prompt(&system, &property, &examples);

        assert!(prompt.contains("IMPLICATION"));
        assert!(prompt.contains("Post-values"));
    }

    #[test]
    fn test_build_prompt_contains_output_format() {
        use kani_fast_kinduction::{SmtType, TransitionSystemBuilder};

        let config = LlmConfig {
            provider: LlmProvider::Mock,
            num_suggestions: 4,
            ..Default::default()
        };
        let client = LlmClient::new(config);

        let system = TransitionSystemBuilder::new()
            .variable("x", SmtType::Int)
            .init("(= x 0)")
            .transition("true")
            .build();

        let property = Property::safety("p1", "test", StateFormula::new("true"));
        let prompt = client.build_prompt(&system, &property, &[]);

        assert!(prompt.contains("Output Format"));
        assert!(prompt.contains("4 invariant suggestions"));
        assert!(prompt.contains("SMT-LIB2"));
    }

    #[test]
    fn test_build_prompt_contains_verification_instructions() {
        use kani_fast_kinduction::{SmtType, TransitionSystemBuilder};

        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        let system = TransitionSystemBuilder::new()
            .variable("n", SmtType::Int)
            .init("(= n 0)")
            .transition("true")
            .build();

        let property = Property::safety("p1", "test", StateFormula::new("true"));
        let prompt = client.build_prompt(&system, &property, &[]);

        assert!(prompt.contains("formal verification"));
        assert!(prompt.contains("initial state"));
        assert!(prompt.contains("transition relation"));
        assert!(prompt.contains("property to be verified"));
    }

    #[test]
    fn test_llm_provider_serialization() {
        let provider = LlmProvider::OpenAI;
        let json = serde_json::to_string(&provider).unwrap();
        let parsed: LlmProvider = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, LlmProvider::OpenAI);

        let provider = LlmProvider::Anthropic;
        let json = serde_json::to_string(&provider).unwrap();
        let parsed: LlmProvider = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, LlmProvider::Anthropic);

        let provider = LlmProvider::Ollama;
        let json = serde_json::to_string(&provider).unwrap();
        let parsed: LlmProvider = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, LlmProvider::Ollama);

        let provider = LlmProvider::Mock;
        let json = serde_json::to_string(&provider).unwrap();
        let parsed: LlmProvider = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, LlmProvider::Mock);
    }

    #[test]
    fn test_parse_invariants_whitespace_handling() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        let response = "```smt\n  (>= x 0)  \n```";
        let invariants = client.parse_invariants(response).unwrap();
        assert_eq!(invariants.len(), 1);
        assert_eq!(invariants[0].smt_formula, "(>= x 0)");
    }

    #[test]
    fn test_parse_invariants_multiple_blocks() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        let response = r"
First invariant:
```smt
(>= x 0)
```

Second invariant:
```smt
(<= x 100)
```

Third invariant:
```smt
(and (>= y 0) (<= y x))
```
";

        let invariants = client.parse_invariants(response).unwrap();
        assert_eq!(invariants.len(), 3);
        assert_eq!(invariants[0].smt_formula, "(>= x 0)");
        assert_eq!(invariants[1].smt_formula, "(<= x 100)");
        assert_eq!(invariants[2].smt_formula, "(and (>= y 0) (<= y x))");
    }

    #[test]
    fn test_parse_invariants_invalid_smt_filtered() {
        let config = LlmConfig::default();
        let client = LlmClient::new(config);

        // Invalid SMT should be filtered out
        let response = r"
```smt
(>= x 0)
```

```smt
x >= 0
```

```smt
(<= y 10)
```
";

        let invariants = client.parse_invariants(response).unwrap();
        // "x >= 0" is not valid SMT (doesn't start with paren), should be filtered
        assert_eq!(invariants.len(), 2);
    }

    #[test]
    fn test_mock_response_empty_system() {
        use kani_fast_kinduction::TransitionSystemBuilder;

        let config = LlmConfig {
            provider: LlmProvider::Mock,
            ..Default::default()
        };
        let client = LlmClient::new(config);

        // System with no integer variables
        let system = TransitionSystemBuilder::new()
            .init("true")
            .transition("true")
            .build();

        let response = client.mock_response(&system);
        // Should be empty or minimal
        assert!(response.is_empty() || !response.contains("(>= "));
    }
}
