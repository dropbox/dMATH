//! Error types for Lake

use std::path::PathBuf;
use thiserror::Error;

/// Result type for Lake operations
pub type LakeResult<T> = Result<T, LakeError>;

/// Error type for Lake operations
#[derive(Debug, Error)]
pub enum LakeError {
    /// Failed to find lakefile.lean
    #[error("lakefile.lean not found in {0}")]
    LakefileNotFound(PathBuf),

    /// Failed to parse lakefile.lean
    #[error("failed to parse lakefile.lean: {0}")]
    LakefileParse(String),

    /// Failed to parse lake-manifest.json
    #[error("failed to parse lake-manifest.json: {0}")]
    ManifestParse(String),

    /// Missing required field
    #[error("missing required field '{field}' in {context}")]
    MissingField { field: String, context: String },

    /// Invalid configuration
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Dependency not found
    #[error("dependency '{name}' not found")]
    DependencyNotFound { name: String },

    /// Dependency does not match manifest entry
    #[error("dependency '{name}' mismatch: {reason}")]
    DependencyMismatch { name: String, reason: String },

    /// Build failed
    #[error("build failed for '{module}': {reason}")]
    BuildFailed { module: String, reason: String },

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Module not found
    #[error("module '{0}' not found")]
    ModuleNotFound(String),

    /// Circular dependency detected
    #[error("circular dependency detected: {0}")]
    CircularDependency(String),

    /// Elaboration error
    #[error("elaboration error: {0}")]
    Elaboration(String),

    /// Type checking error
    #[error("type checking error: {0}")]
    TypeCheck(String),

    /// Git operation failed
    #[error("git {operation} failed: {message}")]
    GitError { operation: String, message: String },

    /// Package not found at expected path
    #[error("package '{name}' not found at {path}")]
    PackageNotFound { name: String, path: PathBuf },

    /// Manifest missing but required for declared dependencies
    #[error("lake-manifest.json missing but dependencies are declared")]
    ManifestMissingForDependencies,
}
