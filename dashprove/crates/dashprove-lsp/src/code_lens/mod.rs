//! Code lens generation for USL documents.
//!
//! Provides clickable action buttons above verifiable properties to trigger verification.
//! Also provides workspace-wide and document-specific statistics lenses for aggregate views.

mod document_stats;
mod lens_generation;
mod types;
mod workspace_stats;

#[cfg(test)]
mod tests;

// Re-export public types
pub use document_stats::{generate_document_stats_lenses, DocumentStats};
pub use lens_generation::generate_all_code_lenses;

// Re-export for tests
#[cfg(test)]
pub(crate) use lens_generation::generate_code_lenses;
pub use types::PropertyCounts;
pub use workspace_stats::{generate_workspace_stats_lenses, WorkspaceStats};
