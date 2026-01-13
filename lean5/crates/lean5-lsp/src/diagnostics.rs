//! Diagnostic generation from parse errors, type errors, and warnings
//!
//! Converts Lean5 parser and elaborator errors/warnings into LSP diagnostics.

use crate::document::{Document, ElaboratedDocument, ParsedDocument, WarningCode};
use tower_lsp::lsp_types::{Diagnostic, DiagnosticSeverity, DiagnosticTag, NumberOrString, Range};

/// Generate LSP diagnostics from a parsed document
#[must_use]
pub fn generate_parse_diagnostics(doc: &Document, parsed: &ParsedDocument) -> Vec<Diagnostic> {
    parsed
        .errors
        .iter()
        .map(|err| {
            let start = doc.offset_to_position(err.start);
            let end = doc.offset_to_position(err.end);

            Diagnostic {
                range: Range { start, end },
                severity: Some(DiagnosticSeverity::ERROR),
                code: Some(NumberOrString::String("parse-error".to_string())),
                code_description: None,
                source: Some("lean5".to_string()),
                message: err.message.clone(),
                related_information: None,
                tags: None,
                data: None,
            }
        })
        .collect()
}

/// Generate LSP diagnostics from an elaborated document's type errors
#[must_use]
pub fn generate_type_diagnostics(doc: &Document, elab: &ElaboratedDocument) -> Vec<Diagnostic> {
    elab.errors
        .iter()
        .map(|err| {
            let start = doc.offset_to_position(err.start);
            let end = doc.offset_to_position(err.end);

            Diagnostic {
                range: Range { start, end },
                severity: Some(DiagnosticSeverity::ERROR),
                code: Some(NumberOrString::String("type-error".to_string())),
                code_description: None,
                source: Some("lean5".to_string()),
                message: err.message.clone(),
                related_information: None,
                tags: None,
                data: None,
            }
        })
        .collect()
}

/// Generate LSP diagnostics from an elaborated document's warnings
#[must_use]
pub fn generate_warning_diagnostics(doc: &Document, elab: &ElaboratedDocument) -> Vec<Diagnostic> {
    elab.warnings
        .iter()
        .map(|warn| {
            let start = doc.offset_to_position(warn.start);
            let end = doc.offset_to_position(warn.end);

            // Map warning codes to diagnostic codes and tags
            let (code_str, tags) = match warn.code {
                WarningCode::UnusedVariable | WarningCode::UnusedImport => {
                    ("unused".to_string(), Some(vec![DiagnosticTag::UNNECESSARY]))
                }
                WarningCode::DeprecatedFeature => (
                    "deprecated".to_string(),
                    Some(vec![DiagnosticTag::DEPRECATED]),
                ),
                WarningCode::UnreachableCode => (
                    "unreachable".to_string(),
                    Some(vec![DiagnosticTag::UNNECESSARY]),
                ),
                WarningCode::IncompleteProof => (
                    "incomplete-proof".to_string(),
                    None, // No special tag - just a warning
                ),
                WarningCode::Other => ("warning".to_string(), None),
            };

            Diagnostic {
                range: Range { start, end },
                severity: Some(DiagnosticSeverity::WARNING),
                code: Some(NumberOrString::String(code_str)),
                code_description: None,
                source: Some("lean5".to_string()),
                message: warn.message.clone(),
                related_information: None,
                tags,
                data: None,
            }
        })
        .collect()
}

/// Combine all diagnostics for a document
#[must_use]
pub fn generate_all_diagnostics(doc: &Document) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    if let Some(parsed) = &doc.parsed {
        diagnostics.extend(generate_parse_diagnostics(doc, parsed));
    }

    if let Some(elab) = &doc.elaborated {
        diagnostics.extend(generate_type_diagnostics(doc, elab));
        diagnostics.extend(generate_warning_diagnostics(doc, elab));
    }

    diagnostics
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{ParseError, TypeError};
    use tower_lsp::lsp_types::Url;

    #[test]
    fn test_parse_diagnostic_generation() {
        let uri = Url::parse("file:///test.lean").unwrap();
        let doc = Document::new(uri, 1, "def x :=\n".to_string(), "lean".to_string());

        let parsed = ParsedDocument {
            errors: vec![ParseError {
                start: 8,
                end: 9,
                message: "expected expression".to_string(),
            }],
            commands: vec![],
        };

        let diagnostics = generate_parse_diagnostics(&doc, &parsed);

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].severity, Some(DiagnosticSeverity::ERROR));
        assert_eq!(diagnostics[0].message, "expected expression");
        assert_eq!(diagnostics[0].source, Some("lean5".to_string()));
    }

    #[test]
    fn test_type_diagnostic_generation() {
        let uri = Url::parse("file:///test.lean").unwrap();
        let doc = Document::new(
            uri,
            1,
            "def x : Nat := \"hello\"\n".to_string(),
            "lean".to_string(),
        );

        let elab = ElaboratedDocument {
            errors: vec![TypeError {
                start: 15,
                end: 22,
                message: "type mismatch: expected Nat, got String".to_string(),
            }],
            warnings: vec![],
            declarations: vec![],
        };

        let diagnostics = generate_type_diagnostics(&doc, &elab);

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].severity, Some(DiagnosticSeverity::ERROR));
        assert!(diagnostics[0].message.contains("type mismatch"));
    }

    #[test]
    fn test_empty_diagnostics() {
        let uri = Url::parse("file:///test.lean").unwrap();
        let doc = Document::new(uri, 1, "def x := 1\n".to_string(), "lean".to_string());

        let parsed = ParsedDocument {
            errors: vec![],
            commands: vec![],
        };

        let diagnostics = generate_parse_diagnostics(&doc, &parsed);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_combined_diagnostics() {
        let uri = Url::parse("file:///test.lean").unwrap();
        let mut doc = Document::new(uri, 1, "def x := 1\n".to_string(), "lean".to_string());

        doc.parsed = Some(ParsedDocument {
            errors: vec![ParseError {
                start: 0,
                end: 3,
                message: "parse error".to_string(),
            }],
            commands: vec![],
        });

        doc.elaborated = Some(ElaboratedDocument {
            errors: vec![TypeError {
                start: 9,
                end: 10,
                message: "type error".to_string(),
            }],
            warnings: vec![],
            declarations: vec![],
        });

        let diagnostics = generate_all_diagnostics(&doc);
        assert_eq!(diagnostics.len(), 2);
    }

    #[test]
    fn test_warning_diagnostic_generation() {
        use crate::document::Warning;

        let uri = Url::parse("file:///test.lean").unwrap();
        let doc = Document::new(
            uri,
            1,
            "def x (unused : Nat) := 1\n".to_string(),
            "lean".to_string(),
        );

        let elab = ElaboratedDocument {
            errors: vec![],
            warnings: vec![Warning {
                start: 7,
                end: 13,
                message: "unused variable 'unused'".to_string(),
                code: WarningCode::UnusedVariable,
            }],
            declarations: vec![],
        };

        let diagnostics = generate_warning_diagnostics(&doc, &elab);

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].severity, Some(DiagnosticSeverity::WARNING));
        assert!(diagnostics[0].message.contains("unused variable"));
        assert_eq!(
            diagnostics[0].code,
            Some(NumberOrString::String("unused".to_string()))
        );
        assert_eq!(diagnostics[0].tags, Some(vec![DiagnosticTag::UNNECESSARY]));
    }

    #[test]
    fn test_deprecated_warning() {
        use crate::document::Warning;

        let uri = Url::parse("file:///test.lean").unwrap();
        let doc = Document::new(
            uri,
            1,
            "def x := oldFunction\n".to_string(),
            "lean".to_string(),
        );

        let elab = ElaboratedDocument {
            errors: vec![],
            warnings: vec![Warning {
                start: 9,
                end: 20,
                message: "'oldFunction' is deprecated".to_string(),
                code: WarningCode::DeprecatedFeature,
            }],
            declarations: vec![],
        };

        let diagnostics = generate_warning_diagnostics(&doc, &elab);

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].severity, Some(DiagnosticSeverity::WARNING));
        assert_eq!(
            diagnostics[0].code,
            Some(NumberOrString::String("deprecated".to_string()))
        );
        assert_eq!(diagnostics[0].tags, Some(vec![DiagnosticTag::DEPRECATED]));
    }

    #[test]
    fn test_combined_errors_and_warnings() {
        use crate::document::Warning;

        let uri = Url::parse("file:///test.lean").unwrap();
        let mut doc = Document::new(
            uri,
            1,
            "def x (y : Nat) := 1\n".to_string(),
            "lean".to_string(),
        );

        doc.parsed = Some(ParsedDocument {
            errors: vec![],
            commands: vec![],
        });

        doc.elaborated = Some(ElaboratedDocument {
            errors: vec![TypeError {
                start: 0,
                end: 3,
                message: "type error".to_string(),
            }],
            warnings: vec![Warning {
                start: 7,
                end: 8,
                message: "unused variable 'y'".to_string(),
                code: WarningCode::UnusedVariable,
            }],
            declarations: vec![],
        });

        let diagnostics = generate_all_diagnostics(&doc);
        assert_eq!(diagnostics.len(), 2);

        // One error, one warning
        let errors: Vec<_> = diagnostics
            .iter()
            .filter(|d| d.severity == Some(DiagnosticSeverity::ERROR))
            .collect();
        let warnings: Vec<_> = diagnostics
            .iter()
            .filter(|d| d.severity == Some(DiagnosticSeverity::WARNING))
            .collect();

        assert_eq!(errors.len(), 1);
        assert_eq!(warnings.len(), 1);
    }
}
