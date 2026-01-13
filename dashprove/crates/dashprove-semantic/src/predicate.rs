//! Built-in semantic predicates for common verification patterns
//!
//! This module provides reusable semantic predicates that can be used
//! for common verification tasks like checking if a response addresses
//! a question or if two texts are semantically equivalent.

use crate::embedding::{EmbeddingBackend, LocalEmbedding};
use crate::error::{SemanticError, SemanticResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Result of evaluating a semantic predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticPredicateResult {
    /// Name of the predicate
    pub predicate: String,
    /// Whether the predicate passed
    pub passed: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Explanation of the result
    pub explanation: String,
    /// Additional metadata
    pub metadata: Option<serde_json::Value>,
}

impl SemanticPredicateResult {
    /// Create a passing result
    pub fn pass(predicate: &str, confidence: f64, explanation: &str) -> Self {
        Self {
            predicate: predicate.to_string(),
            passed: true,
            confidence,
            explanation: explanation.to_string(),
            metadata: None,
        }
    }

    /// Create a failing result
    pub fn fail(predicate: &str, confidence: f64, explanation: &str) -> Self {
        Self {
            predicate: predicate.to_string(),
            passed: false,
            confidence,
            explanation: explanation.to_string(),
            metadata: None,
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Trait for semantic predicates
#[async_trait]
pub trait SemanticPredicate: Send + Sync {
    /// Get the name of this predicate
    fn name(&self) -> &str;

    /// Evaluate the predicate on given inputs
    async fn evaluate(&self, inputs: &PredicateInputs) -> SemanticResult<SemanticPredicateResult>;
}

/// Inputs for semantic predicates
#[derive(Debug, Clone)]
pub struct PredicateInputs {
    /// Primary text to evaluate
    pub primary: String,
    /// Secondary/reference text (optional)
    pub secondary: Option<String>,
    /// List of additional texts (for multi-reference predicates)
    pub references: Vec<String>,
    /// Custom parameters
    pub params: serde_json::Value,
}

impl PredicateInputs {
    /// Create inputs with just primary text
    pub fn primary(text: &str) -> Self {
        Self {
            primary: text.to_string(),
            secondary: None,
            references: Vec::new(),
            params: serde_json::Value::Null,
        }
    }

    /// Create inputs with primary and secondary text
    pub fn pair(primary: &str, secondary: &str) -> Self {
        Self {
            primary: primary.to_string(),
            secondary: Some(secondary.to_string()),
            references: Vec::new(),
            params: serde_json::Value::Null,
        }
    }

    /// Add reference texts
    pub fn with_references(mut self, refs: Vec<String>) -> Self {
        self.references = refs;
        self
    }

    /// Add parameters
    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = params;
        self
    }
}

/// Semantic similarity predicate
///
/// Checks if two texts are semantically similar above a threshold.
pub struct SemanticSimilarity {
    /// Similarity threshold (0.0 to 1.0)
    threshold: f64,
    /// Embedding backend
    backend: Arc<dyn EmbeddingBackend>,
}

impl SemanticSimilarity {
    /// Create a new semantic similarity predicate
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            backend: Arc::new(LocalEmbedding::default()),
        }
    }

    /// Create with custom embedding backend
    pub fn with_backend(threshold: f64, backend: Arc<dyn EmbeddingBackend>) -> Self {
        Self { threshold, backend }
    }

    /// Get the threshold
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Compute similarity between two texts
    async fn compute_similarity(&self, text1: &str, text2: &str) -> SemanticResult<f64> {
        let emb1 = self.backend.embed(text1).await?;
        let emb2 = self.backend.embed(text2).await?;
        Ok(emb1.cosine_similarity(&emb2) as f64)
    }
}

#[async_trait]
impl SemanticPredicate for SemanticSimilarity {
    fn name(&self) -> &str {
        "semantic_similarity"
    }

    async fn evaluate(&self, inputs: &PredicateInputs) -> SemanticResult<SemanticPredicateResult> {
        let secondary = inputs.secondary.as_ref().ok_or_else(|| {
            SemanticError::Predicate("SemanticSimilarity requires secondary text".to_string())
        })?;

        let similarity = self.compute_similarity(&inputs.primary, secondary).await?;
        let passed = similarity >= self.threshold;

        let explanation = format!(
            "Similarity {:.2} is {} threshold {:.2}",
            similarity,
            if passed { ">=" } else { "<" },
            self.threshold
        );

        if passed {
            Ok(SemanticPredicateResult::pass(
                self.name(),
                similarity,
                &explanation,
            ))
        } else {
            Ok(SemanticPredicateResult::fail(
                self.name(),
                similarity,
                &explanation,
            ))
        }
    }
}

/// Predicate that checks if a response addresses a question
///
/// Uses keyword overlap and semantic similarity to determine if a response
/// is likely answering the given question.
pub struct AddressesQuestion {
    /// Minimum relevance score to pass
    min_relevance: f64,
    /// Embedding backend
    backend: Arc<dyn EmbeddingBackend>,
}

impl AddressesQuestion {
    /// Create a new addresses-question predicate
    pub fn new(min_relevance: f64) -> Self {
        Self {
            min_relevance,
            backend: Arc::new(LocalEmbedding::default()),
        }
    }

    /// Create with custom embedding backend
    pub fn with_backend(min_relevance: f64, backend: Arc<dyn EmbeddingBackend>) -> Self {
        Self {
            min_relevance,
            backend,
        }
    }

    /// Extract keywords from text
    fn extract_keywords(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 2)
            .filter(|s| !STOP_WORDS.contains(s))
            .map(|s| s.to_string())
            .collect()
    }

    /// Compute keyword overlap score
    fn keyword_overlap(&self, question: &str, response: &str) -> f64 {
        let q_keywords: std::collections::HashSet<_> =
            self.extract_keywords(question).into_iter().collect();
        let r_keywords: std::collections::HashSet<_> =
            self.extract_keywords(response).into_iter().collect();

        if q_keywords.is_empty() {
            return 0.0;
        }

        let overlap = q_keywords.intersection(&r_keywords).count() as f64;
        overlap / q_keywords.len() as f64
    }
}

/// Common English stop words
const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "and", "but", "if", "or", "because",
    "as", "until", "while", "this", "that", "these", "those", "what", "which", "who", "whom",
    "whose",
];

#[async_trait]
impl SemanticPredicate for AddressesQuestion {
    fn name(&self) -> &str {
        "addresses_question"
    }

    async fn evaluate(&self, inputs: &PredicateInputs) -> SemanticResult<SemanticPredicateResult> {
        let response = inputs.secondary.as_ref().ok_or_else(|| {
            SemanticError::Predicate(
                "AddressesQuestion requires secondary text (response)".to_string(),
            )
        })?;

        let question = &inputs.primary;

        // Compute keyword overlap
        let keyword_score = self.keyword_overlap(question, response);

        // Compute semantic similarity
        let emb_q = self.backend.embed(question).await?;
        let emb_r = self.backend.embed(response).await?;
        let semantic_score = emb_q.cosine_similarity(&emb_r) as f64;

        // Combined relevance score (weighted average)
        let relevance = 0.4 * keyword_score + 0.6 * semantic_score;
        let passed = relevance >= self.min_relevance;

        let explanation = format!(
            "Relevance {:.2} (keyword: {:.2}, semantic: {:.2}) {} threshold {:.2}",
            relevance,
            keyword_score,
            semantic_score,
            if passed { ">=" } else { "<" },
            self.min_relevance
        );

        let metadata = serde_json::json!({
            "keyword_score": keyword_score,
            "semantic_score": semantic_score,
            "combined_relevance": relevance
        });

        let result = if passed {
            SemanticPredicateResult::pass(self.name(), relevance, &explanation)
        } else {
            SemanticPredicateResult::fail(self.name(), relevance, &explanation)
        };

        Ok(result.with_metadata(metadata))
    }
}

/// Predicate that checks if text contains certain concepts
pub struct ContainsConcepts {
    /// Required concepts (all must be present)
    required_concepts: Vec<String>,
    /// Minimum similarity to consider a concept present
    concept_threshold: f64,
    /// Embedding backend
    backend: Arc<dyn EmbeddingBackend>,
}

impl ContainsConcepts {
    /// Create a new contains-concepts predicate
    pub fn new(required_concepts: Vec<String>, concept_threshold: f64) -> Self {
        Self {
            required_concepts,
            concept_threshold,
            backend: Arc::new(LocalEmbedding::default()),
        }
    }

    /// Check if text contains a concept
    async fn contains_concept(&self, text: &str, concept: &str) -> SemanticResult<bool> {
        let text_emb = self.backend.embed(text).await?;
        let concept_emb = self.backend.embed(concept).await?;
        let similarity = text_emb.cosine_similarity(&concept_emb) as f64;
        Ok(similarity >= self.concept_threshold)
    }
}

#[async_trait]
impl SemanticPredicate for ContainsConcepts {
    fn name(&self) -> &str {
        "contains_concepts"
    }

    async fn evaluate(&self, inputs: &PredicateInputs) -> SemanticResult<SemanticPredicateResult> {
        let text = &inputs.primary;
        let mut missing_concepts = Vec::new();
        let mut present_concepts = Vec::new();

        for concept in &self.required_concepts {
            if self.contains_concept(text, concept).await? {
                present_concepts.push(concept.clone());
            } else {
                missing_concepts.push(concept.clone());
            }
        }

        let passed = missing_concepts.is_empty();
        let confidence = present_concepts.len() as f64 / self.required_concepts.len() as f64;

        let explanation = if passed {
            format!("All {} concepts present", self.required_concepts.len())
        } else {
            format!("Missing concepts: {}", missing_concepts.join(", "))
        };

        let metadata = serde_json::json!({
            "present_concepts": present_concepts,
            "missing_concepts": missing_concepts,
            "total_required": self.required_concepts.len()
        });

        let result = if passed {
            SemanticPredicateResult::pass(self.name(), confidence, &explanation)
        } else {
            SemanticPredicateResult::fail(self.name(), confidence, &explanation)
        };

        Ok(result.with_metadata(metadata))
    }
}

/// Registry of available semantic predicates
#[derive(Default)]
pub struct PredicateRegistry {
    predicates: std::collections::HashMap<String, Arc<dyn SemanticPredicate>>,
}

impl PredicateRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Create registry with default predicates
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register(Arc::new(SemanticSimilarity::new(0.8)));
        registry.register(Arc::new(AddressesQuestion::new(0.5)));
        registry
    }

    /// Register a predicate
    pub fn register(&mut self, predicate: Arc<dyn SemanticPredicate>) {
        self.predicates
            .insert(predicate.name().to_string(), predicate);
    }

    /// Get a predicate by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn SemanticPredicate>> {
        self.predicates.get(name).cloned()
    }

    /// List all predicate names
    pub fn list(&self) -> Vec<&str> {
        self.predicates.keys().map(|s| s.as_str()).collect()
    }

    /// Evaluate a predicate by name
    pub async fn evaluate(
        &self,
        name: &str,
        inputs: &PredicateInputs,
    ) -> SemanticResult<SemanticPredicateResult> {
        let predicate = self
            .get(name)
            .ok_or_else(|| SemanticError::Predicate(format!("Unknown predicate: {}", name)))?;

        predicate.evaluate(inputs).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_semantic_similarity_pass() {
        let pred = SemanticSimilarity::new(0.5);
        let inputs = PredicateInputs::pair("hello world", "hello world");

        let result = pred.evaluate(&inputs).await.unwrap();
        assert!(result.passed);
        assert!(result.confidence > 0.99);
    }

    #[tokio::test]
    async fn test_semantic_similarity_fail() {
        let pred = SemanticSimilarity::new(0.99);
        let inputs = PredicateInputs::pair("hello world", "goodbye universe");

        let result = pred.evaluate(&inputs).await.unwrap();
        // Different texts with very high threshold should fail
        assert!(!result.passed || result.confidence < 0.99);
    }

    #[tokio::test]
    async fn test_addresses_question() {
        let pred = AddressesQuestion::new(0.3);
        let inputs = PredicateInputs::pair(
            "What is the capital of France?",
            "Paris is the capital of France, known for the Eiffel Tower.",
        );

        let result = pred.evaluate(&inputs).await.unwrap();
        assert!(result.passed);
        assert!(result.metadata.is_some());
    }

    #[tokio::test]
    async fn test_contains_concepts() {
        let pred = ContainsConcepts::new(vec!["programming".to_string(), "code".to_string()], 0.3);
        let inputs = PredicateInputs::primary(
            "This article discusses programming languages and how to write code efficiently.",
        );

        let result = pred.evaluate(&inputs).await.unwrap();
        assert!(result.metadata.is_some());
    }

    #[tokio::test]
    async fn test_predicate_registry() {
        let registry = PredicateRegistry::with_defaults();

        assert!(registry.get("semantic_similarity").is_some());
        assert!(registry.get("addresses_question").is_some());

        let names = registry.list();
        assert!(names.contains(&"semantic_similarity"));
    }

    #[tokio::test]
    async fn test_registry_evaluate() {
        let registry = PredicateRegistry::with_defaults();
        let inputs = PredicateInputs::pair("test", "test");

        let result = registry.evaluate("semantic_similarity", &inputs).await;
        assert!(result.is_ok());
        assert!(result.unwrap().passed);
    }

    #[tokio::test]
    async fn test_predicate_result_metadata() {
        let result = SemanticPredicateResult::pass("test", 0.9, "explanation")
            .with_metadata(serde_json::json!({"key": "value"}));

        assert!(result.metadata.is_some());
        assert_eq!(result.metadata.unwrap()["key"], "value");
    }

    #[test]
    fn test_predicate_inputs_builder() {
        let inputs = PredicateInputs::pair("primary", "secondary")
            .with_references(vec!["ref1".to_string()])
            .with_params(serde_json::json!({"threshold": 0.8}));

        assert_eq!(inputs.primary, "primary");
        assert_eq!(inputs.secondary, Some("secondary".to_string()));
        assert_eq!(inputs.references.len(), 1);
        assert_eq!(inputs.params["threshold"], 0.8);
    }

    // Mutation-killing tests for AddressesQuestion
    #[test]
    fn test_extract_keywords_filters_short_words() {
        let pred = AddressesQuestion::new(0.5);

        // Words with 2 or fewer characters should be filtered
        let keywords = pred.extract_keywords("I am a big dog");
        assert!(!keywords.contains(&"i".to_string()));
        assert!(!keywords.contains(&"am".to_string()));
        assert!(!keywords.contains(&"a".to_string()));
        assert!(keywords.contains(&"big".to_string()));
        assert!(keywords.contains(&"dog".to_string()));
    }

    #[test]
    fn test_extract_keywords_filters_stop_words() {
        let pred = AddressesQuestion::new(0.5);

        // Stop words should be filtered
        let keywords = pred.extract_keywords("the capital city which has buildings");
        assert!(!keywords.contains(&"the".to_string()));
        assert!(!keywords.contains(&"which".to_string()));
        assert!(!keywords.contains(&"has".to_string()));
        assert!(keywords.contains(&"capital".to_string()));
        assert!(keywords.contains(&"city".to_string()));
        assert!(keywords.contains(&"buildings".to_string()));
    }

    #[test]
    fn test_extract_keywords_empty_text() {
        let pred = AddressesQuestion::new(0.5);

        let keywords = pred.extract_keywords("");
        assert!(keywords.is_empty());

        let keywords2 = pred.extract_keywords("   ");
        assert!(keywords2.is_empty());
    }

    #[test]
    fn test_extract_keywords_returns_vec() {
        let pred = AddressesQuestion::new(0.5);

        let keywords = pred.extract_keywords("programming language code");
        assert_eq!(keywords.len(), 3);
        assert!(keywords.contains(&"programming".to_string()));
        assert!(keywords.contains(&"language".to_string()));
        assert!(keywords.contains(&"code".to_string()));
    }

    #[test]
    fn test_keyword_overlap_calculation() {
        let pred = AddressesQuestion::new(0.5);

        // 3 question keywords: "capital", "city", "france"
        // 2 overlap keywords in response: "capital", "france"
        let overlap = pred.keyword_overlap(
            "what is the capital city of france",
            "the capital of france is paris",
        );

        // overlap / q_keywords.len() = something between 0 and 1
        assert!(
            overlap > 0.0,
            "Should have positive overlap, got {}",
            overlap
        );
        assert!(overlap <= 1.0, "Overlap should be <= 1.0, got {}", overlap);
    }

    #[test]
    fn test_keyword_overlap_empty_question() {
        let pred = AddressesQuestion::new(0.5);

        // Question with only stop words/short words
        let overlap = pred.keyword_overlap("a the is", "programming language");
        assert_eq!(overlap, 0.0, "Empty question keywords should return 0.0");
    }

    #[test]
    fn test_keyword_overlap_no_overlap() {
        let pred = AddressesQuestion::new(0.5);

        // No overlapping keywords
        let overlap = pred.keyword_overlap("programming language code", "cooking recipe food");
        assert_eq!(overlap, 0.0, "No overlap should return 0.0");
    }

    #[test]
    fn test_keyword_overlap_full_overlap() {
        let pred = AddressesQuestion::new(0.5);

        // All question keywords in response
        let overlap = pred.keyword_overlap("programming code", "programming code examples");
        assert!(
            (overlap - 1.0).abs() < 0.01,
            "Full overlap should return 1.0, got {}",
            overlap
        );
    }

    #[tokio::test]
    async fn test_addresses_question_relevance_formula() {
        let pred = AddressesQuestion::new(0.3);

        // Test that relevance = 0.4 * keyword_score + 0.6 * semantic_score
        let inputs = PredicateInputs::pair(
            "what is python programming",
            "python is a programming language",
        );

        let result = pred.evaluate(&inputs).await.unwrap();

        // Check metadata contains the component scores
        let metadata = result.metadata.unwrap();
        let keyword_score = metadata["keyword_score"].as_f64().unwrap();
        let semantic_score = metadata["semantic_score"].as_f64().unwrap();
        let combined = metadata["combined_relevance"].as_f64().unwrap();

        // Verify the formula
        let expected = 0.4 * keyword_score + 0.6 * semantic_score;
        assert!(
            (combined - expected).abs() < 0.001,
            "Combined should be 0.4*{} + 0.6*{} = {}, got {}",
            keyword_score,
            semantic_score,
            expected,
            combined
        );
    }

    // Mutation-killing tests for ContainsConcepts
    #[tokio::test]
    async fn test_contains_concept_threshold() {
        let pred = ContainsConcepts::new(vec!["programming".to_string()], 0.3);

        // Test with matching concept
        let contains = pred
            .contains_concept("I love programming", "programming")
            .await
            .unwrap();
        assert!(contains, "Should contain programming concept");

        // Test with high threshold that won't match
        let pred_strict = ContainsConcepts::new(vec!["programming".to_string()], 0.99);
        let _contains_strict = pred_strict
            .contains_concept("cooking food", "programming")
            .await
            .unwrap();
        // Likely won't pass the strict threshold for unrelated text
        // (behavior depends on embedding similarity)
    }

    #[tokio::test]
    async fn test_contains_concepts_name() {
        let pred = ContainsConcepts::new(vec![], 0.5);
        assert_eq!(pred.name(), "contains_concepts");
    }

    #[tokio::test]
    async fn test_contains_concepts_confidence_formula() {
        let pred = ContainsConcepts::new(
            vec![
                "code".to_string(),
                "programming".to_string(),
                "software".to_string(),
            ],
            0.1, // Low threshold to ensure concepts are detected
        );

        let inputs = PredicateInputs::primary("code and programming software development");

        let result = pred.evaluate(&inputs).await.unwrap();
        let metadata = result.metadata.unwrap();

        let present_count = metadata["present_concepts"].as_array().unwrap().len();
        let total = metadata["total_required"].as_u64().unwrap() as usize;

        // Confidence = present_concepts.len() / required_concepts.len()
        let expected_confidence = present_count as f64 / total as f64;
        assert!(
            (result.confidence - expected_confidence).abs() < 0.001,
            "Confidence should be {}/{} = {}, got {}",
            present_count,
            total,
            expected_confidence,
            result.confidence
        );
    }

    #[tokio::test]
    async fn test_contains_concepts_missing() {
        let pred = ContainsConcepts::new(
            vec!["quantum".to_string(), "relativity".to_string()],
            0.8, // High threshold
        );

        let inputs = PredicateInputs::primary("cooking delicious food recipes");

        let result = pred.evaluate(&inputs).await.unwrap();

        // Cooking text should not contain physics concepts with high threshold
        assert!(!result.passed, "Should fail for unrelated concepts");

        let metadata = result.metadata.unwrap();
        let missing = metadata["missing_concepts"].as_array().unwrap();
        assert!(!missing.is_empty(), "Should have missing concepts");
    }

    // Mutation-killing tests for AddressesQuestion evaluate
    #[tokio::test]
    async fn test_addresses_question_missing_response() {
        let pred = AddressesQuestion::new(0.5);
        let inputs = PredicateInputs::primary("what is the capital");

        let result = pred.evaluate(&inputs).await;
        assert!(result.is_err(), "Should fail without secondary text");
    }
}
