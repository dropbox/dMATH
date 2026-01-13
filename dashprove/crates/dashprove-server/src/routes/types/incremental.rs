use serde::{Deserialize, Serialize};

use crate::routes::types::{BackendIdParam, CompilationResult};

// ============ Incremental Verification Types ============

/// A change to track for incremental verification
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Change {
    /// Type of change
    pub kind: ChangeKind,
    /// Path or identifier of what changed (e.g., file path, function name)
    pub target: String,
    /// Optional: what specifically changed (e.g., function body, type signature)
    pub details: Option<String>,
}

/// Kind of change for incremental verification
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    /// A file was modified
    FileModified,
    /// A file was added
    FileAdded,
    /// A file was deleted
    FileDeleted,
    /// A function/method was modified
    FunctionModified,
    /// A type definition was added
    TypeAdded,
    /// A type definition was modified
    TypeModified,
    /// A dependency was updated
    DependencyChanged,
}

/// Request for incremental verification
#[derive(Debug, Deserialize)]
pub struct IncrementalVerifyRequest {
    /// Base specification (previous state)
    pub base_spec: String,
    /// Current specification (after changes)
    pub current_spec: String,
    /// List of changes that occurred
    pub changes: Vec<Change>,
    /// Optional: specific backend to use
    pub backend: Option<BackendIdParam>,
}

/// Response from incremental verification
#[derive(Debug, Serialize, Deserialize)]
pub struct IncrementalVerifyResponse {
    /// Overall validity
    pub valid: bool,
    /// Number of properties that were cached/reused
    pub cached_count: usize,
    /// Number of properties that were newly verified
    pub verified_count: usize,
    /// Properties that were affected by changes
    pub affected_properties: Vec<String>,
    /// Properties that were unchanged and cached
    pub unchanged_properties: Vec<String>,
    /// Compilation results for affected properties
    pub compilations: Vec<CompilationResult>,
    /// Errors (if any)
    pub errors: Vec<String>,
}
