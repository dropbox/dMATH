//! SelfCheckGPT backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Check method for SelfCheckGPT
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CheckMethod {
    /// BERTScore for semantic similarity
    #[default]
    BertScore,
    /// N-gram overlap
    Ngram,
    /// Natural Language Inference
    NLI,
    /// Prompt-based checking
    Prompt,
    /// Ensemble of methods
    Ensemble,
}

impl CheckMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            CheckMethod::BertScore => "bertscore",
            CheckMethod::Ngram => "ngram",
            CheckMethod::NLI => "nli",
            CheckMethod::Prompt => "prompt",
            CheckMethod::Ensemble => "ensemble",
        }
    }
}

/// Sampling strategy for generating multiple responses
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SamplingStrategy {
    /// Standard sampling
    #[default]
    Standard,
    /// Temperature sampling
    Temperature,
    /// Top-k sampling
    TopK,
    /// Nucleus sampling
    Nucleus,
}

impl SamplingStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            SamplingStrategy::Standard => "standard",
            SamplingStrategy::Temperature => "temperature",
            SamplingStrategy::TopK => "top_k",
            SamplingStrategy::Nucleus => "nucleus",
        }
    }
}

/// SelfCheckGPT backend configuration
#[derive(Debug, Clone)]
pub struct SelfCheckGPTConfig {
    /// Python interpreter path
    pub python_path: Option<PathBuf>,
    /// Check method to use
    pub check_method: CheckMethod,
    /// Sampling strategy
    pub sampling_strategy: SamplingStrategy,
    /// Number of samples to generate
    pub num_samples: usize,
    /// Hallucination threshold (0.0 to 1.0)
    pub hallucination_threshold: f64,
    /// Minimum pass rate threshold (0.0 to 1.0)
    pub pass_rate_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for SelfCheckGPTConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            check_method: CheckMethod::BertScore,
            sampling_strategy: SamplingStrategy::Standard,
            num_samples: 5,
            hallucination_threshold: 0.5,
            pass_rate_threshold: 0.8,
            timeout: Duration::from_secs(180),
        }
    }
}

impl SelfCheckGPTConfig {
    /// Configure for NLI-based checking
    pub fn nli() -> Self {
        Self {
            check_method: CheckMethod::NLI,
            hallucination_threshold: 0.4,
            pass_rate_threshold: 0.85,
            ..Default::default()
        }
    }

    /// Configure for ensemble checking (most accurate)
    pub fn ensemble() -> Self {
        Self {
            check_method: CheckMethod::Ensemble,
            num_samples: 10,
            hallucination_threshold: 0.3,
            pass_rate_threshold: 0.9,
            ..Default::default()
        }
    }

    /// Configure for fast n-gram checking
    pub fn fast_ngram() -> Self {
        Self {
            check_method: CheckMethod::Ngram,
            num_samples: 3,
            hallucination_threshold: 0.6,
            pass_rate_threshold: 0.75,
            ..Default::default()
        }
    }

    /// Configure for strict checking
    pub fn strict() -> Self {
        Self {
            check_method: CheckMethod::BertScore,
            num_samples: 10,
            hallucination_threshold: 0.2,
            pass_rate_threshold: 0.95,
            ..Default::default()
        }
    }
}
