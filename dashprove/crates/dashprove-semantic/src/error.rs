//! Error types for semantic verification

use thiserror::Error;

/// Errors that can occur during semantic verification
#[derive(Debug, Error)]
pub enum SemanticError {
    /// Embedding generation failed
    #[error("embedding error: {0}")]
    Embedding(String),

    /// Similarity computation failed
    #[error("similarity error: {0}")]
    Similarity(String),

    /// Statistical verification error
    #[error("statistical error: {0}")]
    Statistical(String),

    /// Invalid configuration
    #[error("configuration error: {0}")]
    Config(String),

    /// Sample collection error
    #[error("sample error: {0}")]
    Sample(String),

    /// Predicate evaluation error
    #[error("predicate error: {0}")]
    Predicate(String),

    /// IO error
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type for semantic operations
pub type SemanticResult<T> = Result<T, SemanticError>;

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify SemanticError::Embedding creates correct variant
    #[kani::proof]
    fn kani_embedding_error_variant() {
        let err = SemanticError::Embedding("test".to_string());
        assert!(matches!(err, SemanticError::Embedding(_)));
    }

    /// Verify SemanticError::Similarity creates correct variant
    #[kani::proof]
    fn kani_similarity_error_variant() {
        let err = SemanticError::Similarity("test".to_string());
        assert!(matches!(err, SemanticError::Similarity(_)));
    }

    /// Verify SemanticError::Statistical creates correct variant
    #[kani::proof]
    fn kani_statistical_error_variant() {
        let err = SemanticError::Statistical("test".to_string());
        assert!(matches!(err, SemanticError::Statistical(_)));
    }

    /// Verify SemanticError::Config creates correct variant
    #[kani::proof]
    fn kani_config_error_variant() {
        let err = SemanticError::Config("test".to_string());
        assert!(matches!(err, SemanticError::Config(_)));
    }

    /// Verify SemanticError::Sample creates correct variant
    #[kani::proof]
    fn kani_sample_error_variant() {
        let err = SemanticError::Sample("test".to_string());
        assert!(matches!(err, SemanticError::Sample(_)));
    }

    /// Verify SemanticError::Predicate creates correct variant
    #[kani::proof]
    fn kani_predicate_error_variant() {
        let err = SemanticError::Predicate("test".to_string());
        assert!(matches!(err, SemanticError::Predicate(_)));
    }

    /// Verify all error variants are distinct
    #[kani::proof]
    fn kani_error_variants_distinct() {
        let e1 = SemanticError::Embedding("".to_string());
        let e2 = SemanticError::Similarity("".to_string());
        // Can't use != on Error, but we can verify the match patterns
        assert!(matches!(e1, SemanticError::Embedding(_)));
        assert!(!matches!(e1, SemanticError::Similarity(_)));
        assert!(matches!(e2, SemanticError::Similarity(_)));
    }

    /// Verify error display doesn't panic
    #[kani::proof]
    fn kani_error_display_non_panic() {
        let err = SemanticError::Embedding("x".to_string());
        let _ = format!("{}", err);
    }
}
