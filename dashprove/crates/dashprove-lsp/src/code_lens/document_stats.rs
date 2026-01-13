//! Document-level statistics for code lens.
//!
//! Provides statistics about a single USL document and generates
//! code lenses showing document summary, property breakdown, and errors.

use super::types::{format_properties_count, format_types_count, PropertyCounts};
use crate::document::Document;
use crate::symbols::plural;
use tower_lsp::lsp_types::{CodeLens, Command, Position, Range};

/// Document-specific statistics for the current file.
#[derive(Debug, Clone, Default)]
pub struct DocumentStats {
    /// Number of type definitions in this document
    pub type_count: usize,
    /// Property counts by type
    pub properties: PropertyCounts,
    /// Whether document has parse errors
    pub has_parse_error: bool,
    /// Number of type errors
    pub type_error_count: usize,
}

impl DocumentStats {
    /// Create stats from a document.
    pub fn from_document(doc: &Document) -> Self {
        let mut stats = Self::default();

        if doc.parse_error.is_some() {
            stats.has_parse_error = true;
            return stats;
        }

        stats.type_error_count = doc.type_errors.len();

        if let Some(spec) = &doc.spec {
            stats.type_count = spec.types.len();
            stats.properties.add_properties(&spec.properties);
        }

        stats
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

    /// Format a summary for display.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        if self.type_count > 0 {
            parts.push(format_types_count(self.type_count));
        }

        if self.properties.total > 0 {
            parts.push(format_properties_count(self.properties.total));
        }

        if parts.is_empty() {
            if self.has_parse_error {
                "Parse error".to_string()
            } else {
                "Empty file".to_string()
            }
        } else {
            parts.join(", ")
        }
    }

    /// Format detailed breakdown by property type.
    pub fn detailed_breakdown(&self) -> Option<String> {
        self.properties.detailed_breakdown()
    }

    /// Check if the document has any content worth showing stats for.
    pub fn has_content(&self) -> bool {
        self.type_count > 0 || self.properties.total > 0
    }

    /// Get property type variety count.
    pub fn type_variety(&self) -> usize {
        self.properties.type_variety()
    }

    /// Format error status summary.
    pub fn error_summary(&self) -> Option<String> {
        if !self.has_parse_error && self.type_error_count == 0 {
            return None;
        }

        let mut parts = Vec::new();

        if self.has_parse_error {
            parts.push("parse error".to_string());
        }

        if self.type_error_count > 0 {
            parts.push(format!(
                "{} type error{}",
                self.type_error_count,
                plural(self.type_error_count)
            ));
        }

        Some(parts.join(", "))
    }
}

/// Generate document-specific statistics code lenses.
///
/// These lenses appear at the document header (after workspace lenses if present)
/// and show what's in this specific file.
pub fn generate_document_stats_lenses(doc: &Document, line: u32) -> Vec<CodeLens> {
    let mut lenses = Vec::new();
    let stats = DocumentStats::from_document(doc);

    // Only show if document has content
    if !stats.has_content() && !stats.has_parse_error {
        return lenses;
    }

    let range = Range {
        start: Position::new(line, 0),
        end: Position::new(line, 0),
    };

    // Primary lens: Document summary
    lenses.push(CodeLens {
        range,
        command: Some(Command {
            title: format!("📄 This file: {}", stats.summary()),
            command: "dashprove.showDocumentStats".to_string(),
            arguments: Some(vec![serde_json::Value::String(doc.uri.to_string())]),
        }),
        data: None,
    });

    // Secondary lens: Property breakdown (if multiple property types)
    if let Some(breakdown) = stats.detailed_breakdown() {
        // Only show breakdown if there are multiple property types or more than 2 properties
        if stats.type_variety() > 1 || stats.property_count() > 2 {
            lenses.push(CodeLens {
                range,
                command: Some(Command {
                    title: format!("📋 {}", breakdown),
                    command: "dashprove.showDocumentBreakdown".to_string(),
                    arguments: Some(vec![serde_json::Value::String(doc.uri.to_string())]),
                }),
                data: None,
            });
        }
    }

    // Tertiary lens: Verify all in this file (if there are properties)
    if stats.property_count() > 0 {
        lenses.push(CodeLens {
            range,
            command: Some(Command {
                title: format!("▶ Verify {} in this file", stats.property_count()),
                command: "dashprove.verifyDocument".to_string(),
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
                command: "dashprove.showDocumentErrors".to_string(),
                arguments: Some(vec![serde_json::Value::String(doc.uri.to_string())]),
            }),
            data: None,
        });
    }

    lenses
}
