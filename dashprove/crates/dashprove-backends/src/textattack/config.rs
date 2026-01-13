//! TextAttack backend configuration

use std::path::PathBuf;
use std::time::Duration;

/// Attack recipe for TextAttack NLP adversarial evaluation
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[allow(clippy::upper_case_acronyms)]
pub enum TextAttackRecipe {
    /// TextFooler: word substitution attack (Jin et al., 2019)
    #[default]
    TextFooler,
    /// BERT-Attack: BERT-based word substitution (Li et al., 2020)
    BertAttack,
    /// BAE: BERT-based adversarial examples (Garg & Ramakrishnan, 2020)
    BAE,
    /// DeepWordBug: character-level perturbations (Gao et al., 2018)
    DeepWordBug,
    /// TextBugger: hybrid attack (Li et al., 2018)
    TextBugger,
    /// PWWS: Probability Weighted Word Saliency (Ren et al., 2019)
    PWWS,
    /// Checklist: linguistic template attack
    CheckList,
    /// A2T: Attack to Training (Yoo et al., 2021)
    A2T,
    /// Clare: Contextualized perturbation (Li et al., 2020)
    Clare,
}

impl TextAttackRecipe {
    /// Get the TextAttack recipe name
    pub fn recipe_name(&self) -> &'static str {
        match self {
            TextAttackRecipe::TextFooler => "textfooler",
            TextAttackRecipe::BertAttack => "bert-attack",
            TextAttackRecipe::BAE => "bae",
            TextAttackRecipe::DeepWordBug => "deepwordbug",
            TextAttackRecipe::TextBugger => "textbugger",
            TextAttackRecipe::PWWS => "pwws",
            TextAttackRecipe::CheckList => "checklist",
            TextAttackRecipe::A2T => "a2t",
            TextAttackRecipe::Clare => "clare",
        }
    }

    /// Get a description of the attack
    pub fn description(&self) -> &'static str {
        match self {
            TextAttackRecipe::TextFooler => "Word substitution using counter-fitted embeddings",
            TextAttackRecipe::BertAttack => "BERT-based contextual word substitution",
            TextAttackRecipe::BAE => "BERT-based adversarial examples with masking",
            TextAttackRecipe::DeepWordBug => "Character-level perturbations (typos)",
            TextAttackRecipe::TextBugger => "Hybrid word and character perturbations",
            TextAttackRecipe::PWWS => "Word saliency-based substitution",
            TextAttackRecipe::CheckList => "Linguistic template-based attack",
            TextAttackRecipe::A2T => "Adversarial training-aware attack",
            TextAttackRecipe::Clare => "Contextualized perturbation attack",
        }
    }
}

/// TextAttack backend configuration
#[derive(Debug, Clone)]
pub struct TextAttackConfig {
    /// Path to Python interpreter
    pub python_path: Option<PathBuf>,
    /// Attack recipe to use
    pub attack_recipe: TextAttackRecipe,
    /// Number of samples to attack
    pub num_examples: usize,
    /// Maximum percentage of words to perturb (0.0 to 1.0)
    pub max_percent_words: f64,
    /// Verification timeout
    pub timeout: Duration,
    /// Model name or path (HuggingFace model identifier)
    pub model_name: Option<String>,
    /// Dataset name (HuggingFace dataset identifier)
    pub dataset_name: Option<String>,
    /// Enable query-efficient mode (fewer model queries)
    pub query_efficient: bool,
    /// Minimum semantic similarity threshold (0.0 to 1.0)
    pub min_similarity: f64,
}

impl Default for TextAttackConfig {
    fn default() -> Self {
        Self {
            python_path: None,
            attack_recipe: TextAttackRecipe::TextFooler,
            num_examples: 100,
            max_percent_words: 0.2,
            timeout: Duration::from_secs(600),
            model_name: None,
            dataset_name: None,
            query_efficient: false,
            min_similarity: 0.8,
        }
    }
}

impl TextAttackConfig {
    /// Create config with BERT-Attack recipe
    pub fn bert_attack() -> Self {
        Self {
            attack_recipe: TextAttackRecipe::BertAttack,
            min_similarity: 0.85,
            ..Default::default()
        }
    }

    /// Create config with DeepWordBug (character-level)
    pub fn deep_word_bug() -> Self {
        Self {
            attack_recipe: TextAttackRecipe::DeepWordBug,
            max_percent_words: 0.3,
            ..Default::default()
        }
    }

    /// Create config with PWWS recipe
    pub fn pwws() -> Self {
        Self {
            attack_recipe: TextAttackRecipe::PWWS,
            ..Default::default()
        }
    }

    /// Create config for sentiment analysis
    pub fn for_sentiment() -> Self {
        Self {
            model_name: Some("textattack/bert-base-uncased-SST-2".to_string()),
            dataset_name: Some("sst2".to_string()),
            ..Default::default()
        }
    }
}
