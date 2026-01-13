//! FactScore backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Knowledge source for fact verification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KnowledgeSource {
    /// Wikipedia knowledge base
    #[default]
    Wikipedia,
    /// Custom knowledge base
    Custom,
    /// Web search
    WebSearch,
    /// Retrieved documents
    Retrieved,
}

impl KnowledgeSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            KnowledgeSource::Wikipedia => "wikipedia",
            KnowledgeSource::Custom => "custom",
            KnowledgeSource::WebSearch => "web_search",
            KnowledgeSource::Retrieved => "retrieved",
        }
    }
}

/// Fact extraction method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExtractionMethod {
    /// Sentence-level extraction
    #[default]
    Sentence,
    /// Claim-level extraction
    Claim,
    /// Entity-level extraction
    Entity,
    /// Triple extraction (subject-predicate-object)
    Triple,
}

impl ExtractionMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExtractionMethod::Sentence => "sentence",
            ExtractionMethod::Claim => "claim",
            ExtractionMethod::Entity => "entity",
            ExtractionMethod::Triple => "triple",
        }
    }
}

/// FactScore backend configuration
#[derive(Debug, Clone)]
pub struct FactScoreConfig {
    /// Python interpreter path
    pub python_path: Option<PathBuf>,
    /// Knowledge source for verification
    pub knowledge_source: KnowledgeSource,
    /// Fact extraction method
    pub extraction_method: ExtractionMethod,
    /// Enable atomic fact decomposition
    pub atomic_facts: bool,
    /// Minimum confidence for fact support (0.0 to 1.0)
    pub confidence_threshold: f64,
    /// Minimum pass rate threshold (0.0 to 1.0)
    pub pass_rate_threshold: f64,
    /// Timeout for verification
    pub timeout: Duration,
}

impl Default for FactScoreConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            knowledge_source: KnowledgeSource::Wikipedia,
            extraction_method: ExtractionMethod::Sentence,
            atomic_facts: true,
            confidence_threshold: 0.7,
            pass_rate_threshold: 0.8,
            timeout: Duration::from_secs(180),
        }
    }
}

impl FactScoreConfig {
    /// Configure for claim-level verification
    pub fn claim_level() -> Self {
        Self {
            extraction_method: ExtractionMethod::Claim,
            atomic_facts: true,
            confidence_threshold: 0.75,
            pass_rate_threshold: 0.85,
            ..Default::default()
        }
    }

    /// Configure for strict verification
    pub fn strict() -> Self {
        Self {
            extraction_method: ExtractionMethod::Claim,
            atomic_facts: true,
            confidence_threshold: 0.9,
            pass_rate_threshold: 0.95,
            ..Default::default()
        }
    }

    /// Configure for entity-focused verification
    pub fn entity_focused() -> Self {
        Self {
            extraction_method: ExtractionMethod::Entity,
            confidence_threshold: 0.8,
            pass_rate_threshold: 0.85,
            ..Default::default()
        }
    }

    /// Configure for triple extraction (knowledge graph style)
    pub fn knowledge_graph() -> Self {
        Self {
            extraction_method: ExtractionMethod::Triple,
            atomic_facts: false,
            confidence_threshold: 0.75,
            pass_rate_threshold: 0.8,
            ..Default::default()
        }
    }
}
