//! AI-assisted invariant synthesis for Kani Fast
//!
//! This crate provides AI-powered invariant synthesis when template-based
//! and CHC-based methods fail to find adequate loop invariants.
//!
//! # Components
//!
//! - **ICE Learning**: Iterative Counterexample-guided Inductive synthesis
//! - **LLM Integration**: Query language models for invariant suggestions
//! - **Invariant Corpus**: Store and retrieve successful invariants
//!
//! # Example
//!
//! ```ignore
//! use kani_fast_ai::{AiSynthesizer, IceEngine};
//!
//! let synthesizer = AiSynthesizer::new();
//! let invariant = synthesizer.synthesize(&system, &property).await?;
//! ```

pub mod corpus;
pub mod ice;
pub mod llm;

pub use corpus::{CorpusConfig, InvariantCorpus, InvariantEntry};
pub use ice::{Example, ExampleKind, IceConfig, IceEngine, IceResult};
pub use llm::{LlmClient, LlmConfig, LlmProvider};

use kani_fast_kinduction::{Property, StateFormula, TransitionSystem};
use thiserror::Error;

/// Errors from AI synthesis
#[derive(Debug, Error)]
pub enum AiError {
    #[error("LLM API error: {0}")]
    LlmApi(String),

    #[error("ICE learning failed: {0}")]
    IceLearning(String),

    #[error("Corpus error: {0}")]
    Corpus(String),

    #[error("Synthesis timeout")]
    Timeout,

    #[error("No invariant found after {0} attempts")]
    ExhaustedAttempts(usize),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Database error: {0}")]
    Database(String),
}

/// Configuration for AI synthesis
#[derive(Debug, Clone)]
pub struct AiConfig {
    /// Maximum synthesis attempts
    pub max_attempts: usize,
    /// Timeout per synthesis attempt (seconds)
    pub timeout_secs: u64,
    /// Use LLM for suggestions
    pub use_llm: bool,
    /// Use invariant corpus for lookup
    pub use_corpus: bool,
    /// LLM configuration
    pub llm: LlmConfig,
    /// Corpus configuration
    pub corpus: CorpusConfig,
    /// ICE configuration
    pub ice: IceConfig,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            max_attempts: 10,
            timeout_secs: 30,
            use_llm: true,
            use_corpus: true,
            llm: LlmConfig::default(),
            corpus: CorpusConfig::default(),
            ice: IceConfig::default(),
        }
    }
}

/// AI-powered invariant synthesizer
pub struct AiSynthesizer {
    config: AiConfig,
    corpus: Option<InvariantCorpus>,
    llm: Option<LlmClient>,
    ice: IceEngine,
}

impl AiSynthesizer {
    /// Create a new AI synthesizer with default configuration
    pub fn new() -> Self {
        Self::with_config(AiConfig::default())
    }

    /// Create a new AI synthesizer with custom configuration
    pub fn with_config(config: AiConfig) -> Self {
        let corpus = if config.use_corpus {
            InvariantCorpus::open_default().ok()
        } else {
            None
        };

        let llm = if config.use_llm {
            Some(LlmClient::new(config.llm.clone()))
        } else {
            None
        };

        let ice = IceEngine::new(config.ice.clone());

        Self {
            config,
            corpus,
            llm,
            ice,
        }
    }

    /// Synthesize an invariant for the given system and property
    ///
    /// Strategy:
    /// 1. Check corpus for similar problems
    /// 2. Run ICE learning with templates
    /// 3. If stuck, query LLM for suggestions
    /// 4. Verify suggestions with ICE
    /// 5. Store successful invariants in corpus
    pub async fn synthesize(
        &mut self,
        system: &TransitionSystem,
        property: &Property,
    ) -> Result<SynthesisResult, AiError> {
        let mut attempts = 0;

        // Step 1: Check corpus for similar problems
        if let Some(corpus) = &self.corpus {
            if let Some(entry) = corpus.find_similar(system, property)? {
                tracing::info!("Found similar invariant in corpus: {:?}", entry.invariant);
                // Verify it works for this system
                if self
                    .ice
                    .verify_invariant(&entry.invariant, system, property)?
                {
                    return Ok(SynthesisResult {
                        invariant: entry.invariant.clone(),
                        source: InvariantSource::Corpus(entry.id),
                        attempts: 1,
                        examples_collected: 0,
                    });
                }
            }
        }

        // Step 2: Run ICE learning
        let ice_result = self.ice.learn(system, property)?;
        attempts += ice_result.iterations;

        if let Some(invariant) = ice_result.invariant {
            // Store in corpus
            if let Some(ref mut corpus) = self.corpus {
                let _ = corpus.store(system, property, &invariant);
            }
            return Ok(SynthesisResult {
                invariant,
                source: InvariantSource::Ice,
                attempts,
                examples_collected: ice_result.examples.len(),
            });
        }

        // Step 3: Query LLM for suggestions
        if let Some(llm) = &self.llm {
            while attempts < self.config.max_attempts {
                attempts += 1;

                // Build prompt with ICE examples
                let suggestions = llm
                    .suggest_invariants(system, property, &ice_result.examples)
                    .await?;

                for suggestion in suggestions {
                    // Verify with ICE
                    if self.ice.verify_invariant(&suggestion, system, property)? {
                        // Store in corpus
                        if let Some(ref mut corpus) = self.corpus {
                            let _ = corpus.store(system, property, &suggestion);
                        }
                        return Ok(SynthesisResult {
                            invariant: suggestion,
                            source: InvariantSource::Llm,
                            attempts,
                            examples_collected: ice_result.examples.len(),
                        });
                    }

                    // Add counterexample to ICE
                    if let Some(cex) = self.ice.get_counterexample(&suggestion, system, property)? {
                        self.ice.add_example(cex);
                    }
                }
            }
        }

        Err(AiError::ExhaustedAttempts(attempts))
    }

    /// Get the invariant corpus
    pub fn corpus(&self) -> Option<&InvariantCorpus> {
        self.corpus.as_ref()
    }

    /// Get mutable access to the invariant corpus
    pub fn corpus_mut(&mut self) -> Option<&mut InvariantCorpus> {
        self.corpus.as_mut()
    }
}

impl Default for AiSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of AI synthesis
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// The discovered invariant
    pub invariant: StateFormula,
    /// Source of the invariant
    pub source: InvariantSource,
    /// Number of synthesis attempts
    pub attempts: usize,
    /// Number of examples collected during ICE
    pub examples_collected: usize,
}

/// Source of synthesized invariant
#[derive(Debug, Clone)]
pub enum InvariantSource {
    /// Found in corpus
    Corpus(i64),
    /// Discovered via ICE learning
    Ice,
    /// Suggested by LLM
    Llm,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_config_default() {
        let config = AiConfig::default();
        assert_eq!(config.max_attempts, 10);
        assert!(config.use_llm);
        assert!(config.use_corpus);
    }

    #[test]
    fn test_ai_synthesizer_creation() {
        let config = AiConfig {
            use_llm: false,
            use_corpus: false,
            ..Default::default()
        };
        let synth = AiSynthesizer::with_config(config);
        assert!(synth.corpus.is_none());
        assert!(synth.llm.is_none());
    }

    // =============================================================================
    // Mutation Coverage Tests (targeting specific mutants)
    // =============================================================================

    #[test]
    fn test_ai_synthesizer_corpus_accessor() {
        // Test with corpus disabled - should return None
        let config = AiConfig {
            use_corpus: false,
            use_llm: false,
            ..Default::default()
        };
        let synth = AiSynthesizer::with_config(config);
        assert!(
            synth.corpus().is_none(),
            "corpus() should return None when disabled"
        );

        // Test with corpus enabled - should return Some
        // Note: We can't actually test this because opening the default corpus
        // may fail in test environments. Instead, we just verify the method works.
    }

    #[test]
    fn test_ai_synthesizer_corpus_mut_accessor() {
        // Test with corpus disabled - should return None
        let config = AiConfig {
            use_corpus: false,
            use_llm: false,
            ..Default::default()
        };
        let mut synth = AiSynthesizer::with_config(config);
        assert!(
            synth.corpus_mut().is_none(),
            "corpus_mut() should return None when disabled"
        );
    }

    #[test]
    fn test_ai_config_timeout() {
        let config = AiConfig::default();
        assert_eq!(config.timeout_secs, 30);

        let custom = AiConfig {
            timeout_secs: 60,
            ..Default::default()
        };
        assert_eq!(custom.timeout_secs, 60);
    }

    #[test]
    fn test_synthesis_result_fields() {
        let result = SynthesisResult {
            invariant: StateFormula::new("(>= x 0)"),
            source: InvariantSource::Ice,
            attempts: 5,
            examples_collected: 10,
        };

        assert_eq!(result.attempts, 5);
        assert_eq!(result.examples_collected, 10);
        assert!(matches!(result.source, InvariantSource::Ice));
    }

    #[test]
    fn test_synthesis_result_clone() {
        let result = SynthesisResult {
            invariant: StateFormula::new("(>= x 0)"),
            source: InvariantSource::Corpus(42),
            attempts: 3,
            examples_collected: 5,
        };

        let cloned = result.clone();
        assert_eq!(cloned.attempts, 3);
        assert_eq!(cloned.examples_collected, 5);
        if let InvariantSource::Corpus(id) = cloned.source {
            assert_eq!(id, 42);
        } else {
            panic!("Expected Corpus source");
        }
    }

    #[test]
    fn test_invariant_source_variants() {
        let corpus_src = InvariantSource::Corpus(123);
        let ice_src = InvariantSource::Ice;
        let llm_src = InvariantSource::Llm;

        // Test Debug
        let _ = format!("{:?}", corpus_src);
        let _ = format!("{:?}", ice_src);
        let _ = format!("{:?}", llm_src);

        // Test Clone
        let cloned = corpus_src.clone();
        assert!(matches!(cloned, InvariantSource::Corpus(123)));
    }

    #[test]
    fn test_ai_error_display() {
        let err1 = AiError::ExhaustedAttempts(5);
        assert!(err1.to_string().contains("5"));

        let err2 = AiError::LlmApi("test error".to_string());
        assert!(err2.to_string().contains("test error"));

        let err3 = AiError::Timeout;
        assert!(err3.to_string().contains("timeout"));
    }

    #[test]
    fn test_ai_synthesizer_default() {
        // Test Default trait implementation
        let synth = AiSynthesizer::default();
        // Default should have LLM and corpus enabled (though corpus may fail to open)
        // Just verify it doesn't panic
        let _ = synth;
    }

    #[test]
    fn test_ai_config_clone() {
        let config = AiConfig {
            max_attempts: 20,
            timeout_secs: 45,
            use_llm: false,
            use_corpus: true,
            ..Default::default()
        };

        let cloned = config.clone();
        assert_eq!(cloned.max_attempts, 20);
        assert_eq!(cloned.timeout_secs, 45);
        assert!(!cloned.use_llm);
        assert!(cloned.use_corpus);
    }

    #[test]
    fn test_ai_config_debug() {
        let config = AiConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("max_attempts"));
        assert!(debug_str.contains("10")); // default max_attempts
    }
}
