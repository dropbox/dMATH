//! Workspace-level statistics for code lens.
//!
//! Provides aggregate statistics across all open USL documents and generates
//! code lenses showing workspace summary, property breakdown, and errors.

use super::types::{
    format_files_count, format_properties_count, format_types_count, PropertyCounts,
};
use crate::document::Document;
use crate::symbols::plural;
use tower_lsp::lsp_types::{CodeLens, Command, Position, Range};

/// Workspace-wide statistics collected from all open documents.
#[derive(Debug, Clone, Default)]
pub struct WorkspaceStats {
    /// Total number of USL files
    pub file_count: usize,
    /// Total number of type definitions
    pub type_count: usize,
    /// Property counts by type
    pub properties: PropertyCounts,
    /// Count of files with parse errors
    pub files_with_errors: usize,
    /// Count of files with type errors
    pub files_with_type_errors: usize,
}

impl WorkspaceStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add statistics from a single document.
    pub fn add_document(&mut self, doc: &Document) {
        self.file_count += 1;

        if doc.parse_error.is_some() {
            self.files_with_errors += 1;
            return;
        }

        if !doc.type_errors.is_empty() {
            self.files_with_type_errors += 1;
        }

        if let Some(spec) = &doc.spec {
            self.type_count += spec.types.len();
            self.properties.add_properties(&spec.properties);
        }
    }

    /// Get total property count for convenience.
    pub fn property_count(&self) -> usize {
        self.properties.total
    }

    /// Get theorem count for convenience.
    pub fn theorem_count(&self) -> usize {
        self.properties.theorems
    }

    /// Get contract count for convenience.
    pub fn contract_count(&self) -> usize {
        self.properties.contracts
    }

    /// Get temporal count for convenience.
    pub fn temporal_count(&self) -> usize {
        self.properties.temporal
    }

    /// Get invariant count for convenience.
    pub fn invariant_count(&self) -> usize {
        self.properties.invariants
    }

    /// Get refinement count for convenience.
    pub fn refinement_count(&self) -> usize {
        self.properties.refinements
    }

    /// Get probabilistic count for convenience.
    pub fn probabilistic_count(&self) -> usize {
        self.properties.probabilistic
    }

    /// Get security count for convenience.
    pub fn security_count(&self) -> usize {
        self.properties.security
    }

    /// Format the stats summary for display in code lens.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        if self.file_count > 0 {
            parts.push(format_files_count(self.file_count));
        }

        if self.type_count > 0 {
            parts.push(format_types_count(self.type_count));
        }

        if self.properties.total > 0 {
            parts.push(format_properties_count(self.properties.total));
        }

        if parts.is_empty() {
            "No USL definitions".to_string()
        } else {
            parts.join(", ")
        }
    }

    /// Format detailed breakdown by property type.
    pub fn detailed_breakdown(&self) -> String {
        self.properties
            .detailed_breakdown()
            .unwrap_or_else(|| "No properties".to_string())
    }

    /// Format error status summary.
    pub fn error_summary(&self) -> Option<String> {
        if self.files_with_errors == 0 && self.files_with_type_errors == 0 {
            return None;
        }

        let mut parts = Vec::new();

        if self.files_with_errors > 0 {
            parts.push(format!(
                "{} parse error{}",
                self.files_with_errors,
                plural(self.files_with_errors)
            ));
        }

        if self.files_with_type_errors > 0 {
            parts.push(format!(
                "{} type error{}",
                self.files_with_type_errors,
                plural(self.files_with_type_errors)
            ));
        }

        Some(parts.join(", "))
    }
}

/// Generate workspace statistics code lenses for the top of a document.
///
/// These lenses appear at line 0 and show aggregate statistics across
/// all open USL files in the workspace.
pub fn generate_workspace_stats_lenses(doc: &Document, stats: &WorkspaceStats) -> Vec<CodeLens> {
    let mut lenses = Vec::new();

    // Only show workspace lenses if there are multiple files or significant content
    if stats.file_count < 2 && stats.property_count() == 0 && stats.type_count == 0 {
        return lenses;
    }

    let range = Range {
        start: Position::new(0, 0),
        end: Position::new(0, 0),
    };

    // Primary lens: Workspace summary
    lenses.push(CodeLens {
        range,
        command: Some(Command {
            title: format!("📊 Workspace: {}", stats.summary()),
            command: "dashprove.showWorkspaceStats".to_string(),
            arguments: Some(vec![serde_json::Value::String(doc.uri.to_string())]),
        }),
        data: None,
    });

    // Secondary lens: Property breakdown (if there are properties)
    if stats.property_count() > 0 {
        lenses.push(CodeLens {
            range,
            command: Some(Command {
                title: format!("📋 {}", stats.detailed_breakdown()),
                command: "dashprove.showPropertyBreakdown".to_string(),
                arguments: Some(vec![serde_json::Value::String(doc.uri.to_string())]),
            }),
            data: None,
        });

        // Tertiary lens: Verify all properties
        lenses.push(CodeLens {
            range,
            command: Some(Command {
                title: format!("▶▶ Verify all {} properties", stats.property_count()),
                command: "dashprove.verifyAll".to_string(),
                arguments: Some(vec![serde_json::Value::String(doc.uri.to_string())]),
            }),
            data: None,
        });
    }

    // Error lens (if there are errors)
    if let Some(error_summary) = stats.error_summary() {
        lenses.push(CodeLens {
            range,
            command: Some(Command {
                title: format!("⚠️ {}", error_summary),
                command: "dashprove.showErrors".to_string(),
                arguments: Some(vec![serde_json::Value::String(doc.uri.to_string())]),
            }),
            data: None,
        });
    }

    lenses
}
