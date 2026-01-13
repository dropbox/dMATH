//! Beautiful error rendering using ariadne
//!
//! This module provides rich, colorful diagnostic output for TLA+ errors.
//! It converts various error types to ariadne Reports for display.

use crate::span::Span;
use ariadne::{Color, ColorGenerator, Label, Report, ReportKind, Source};
use std::collections::HashMap;
use std::io::Write;

/// A source file cache for ariadne rendering
pub struct SourceCache {
    /// Map from file path to source content
    files: HashMap<String, Source<String>>,
}

impl SourceCache {
    /// Create a new empty source cache
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    /// Add a source file to the cache
    pub fn add(&mut self, path: impl Into<String>, source: impl Into<String>) {
        let path = path.into();
        let source = source.into();
        self.files.insert(path, Source::from(source));
    }

    /// Get a source from the cache
    pub fn get(&self, path: &str) -> Option<&Source<String>> {
        self.files.get(path)
    }
}

impl Default for SourceCache {
    fn default() -> Self {
        Self::new()
    }
}

/// A diagnostic that can be rendered with ariadne
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// The severity of this diagnostic
    pub severity: Severity,
    /// The main error message
    pub message: String,
    /// The primary span (highlighted in red)
    pub span: Option<DiagnosticSpan>,
    /// Additional labels (notes, hints, related locations)
    pub labels: Vec<DiagnosticLabel>,
    /// Help text shown at the bottom
    pub help: Option<String>,
    /// Note text shown at the bottom
    pub note: Option<String>,
}

/// Diagnostic severity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// A span for diagnostic display
#[derive(Debug, Clone)]
pub struct DiagnosticSpan {
    /// File path
    pub file: String,
    /// Start byte offset
    pub start: usize,
    /// End byte offset
    pub end: usize,
    /// Optional label text
    pub label: Option<String>,
}

/// An additional label on a diagnostic
#[derive(Debug, Clone)]
pub struct DiagnosticLabel {
    /// File path
    pub file: String,
    /// Start byte offset
    pub start: usize,
    /// End byte offset
    pub end: usize,
    /// Label text
    pub text: String,
    /// Label color
    pub color: LabelColor,
}

/// Color for a diagnostic label
#[derive(Debug, Clone, Copy)]
pub enum LabelColor {
    Primary,
    Secondary,
    Info,
}

impl Diagnostic {
    /// Create a new error diagnostic
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            message: message.into(),
            span: None,
            labels: Vec::new(),
            help: None,
            note: None,
        }
    }

    /// Create a new warning diagnostic
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
            span: None,
            labels: Vec::new(),
            help: None,
            note: None,
        }
    }

    /// Set the primary span
    pub fn with_span(mut self, file: impl Into<String>, start: usize, end: usize) -> Self {
        self.span = Some(DiagnosticSpan {
            file: file.into(),
            start,
            end,
            label: None,
        });
        self
    }

    /// Set the primary span with a label
    pub fn with_span_label(
        mut self,
        file: impl Into<String>,
        start: usize,
        end: usize,
        label: impl Into<String>,
    ) -> Self {
        self.span = Some(DiagnosticSpan {
            file: file.into(),
            start,
            end,
            label: Some(label.into()),
        });
        self
    }

    /// Add a secondary label
    pub fn with_label(
        mut self,
        file: impl Into<String>,
        start: usize,
        end: usize,
        text: impl Into<String>,
    ) -> Self {
        self.labels.push(DiagnosticLabel {
            file: file.into(),
            start,
            end,
            text: text.into(),
            color: LabelColor::Secondary,
        });
        self
    }

    /// Add help text
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Add a note
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// Render this diagnostic to a writer
    pub fn render(
        &self,
        file_path: &str,
        source: &str,
        writer: &mut impl Write,
    ) -> std::io::Result<()> {
        let kind = match self.severity {
            Severity::Error => ReportKind::Error,
            Severity::Warning => ReportKind::Warning,
            Severity::Info => ReportKind::Advice,
        };

        let primary_offset = self.span.as_ref().map(|s| s.start).unwrap_or(0);

        let mut builder =
            Report::build(kind, file_path, primary_offset).with_message(&self.message);

        let mut colors = ColorGenerator::new();

        // Add the primary span label
        if let Some(ref span) = self.span {
            let label = Label::new((file_path, span.start..span.end))
                .with_color(Color::Red)
                .with_order(0);
            let label = if let Some(ref text) = span.label {
                label.with_message(text)
            } else {
                label
            };
            builder = builder.with_label(label);
        }

        // Add secondary labels
        for (i, lab) in self.labels.iter().enumerate() {
            let color = match lab.color {
                LabelColor::Primary => Color::Red,
                LabelColor::Secondary => colors.next(),
                LabelColor::Info => Color::Cyan,
            };
            builder = builder.with_label(
                Label::new((file_path, lab.start..lab.end))
                    .with_color(color)
                    .with_message(&lab.text)
                    .with_order((i + 1) as i32),
            );
        }

        // Add help text
        if let Some(ref help) = self.help {
            builder = builder.with_help(help);
        }

        // Add note
        if let Some(ref note) = self.note {
            builder = builder.with_note(note);
        }

        let report = builder.finish();

        // Write with the source
        report.write((file_path, Source::from(source)), writer)
    }

    /// Render this diagnostic to stderr
    pub fn eprint(&self, file_path: &str, source: &str) {
        let mut buf = Vec::new();
        let _ = self.render(file_path, source, &mut buf);
        let _ = std::io::stderr().write_all(&buf);
    }
}

/// Trait for types that can be converted to diagnostics
pub trait IntoDiagnostic {
    /// Convert this error into a diagnostic
    fn into_diagnostic(self, file_path: &str) -> Diagnostic;
}

/// Create a diagnostic from a parse error
pub fn parse_error_diagnostic(file_path: &str, message: &str, start: u32, end: u32) -> Diagnostic {
    Diagnostic::error(format!("syntax error: {}", message)).with_span_label(
        file_path,
        start as usize,
        end as usize,
        "here",
    )
}

/// Create a diagnostic from a lower error
pub fn lower_error_diagnostic(file_path: &str, message: &str, span: Span) -> Diagnostic {
    Diagnostic::error(format!("semantic error: {}", message)).with_span_label(
        file_path,
        span.start as usize,
        span.end as usize,
        "here",
    )
}

/// Create a diagnostic from an undefined name error
pub fn undefined_name_diagnostic(file_path: &str, name: &str, span: Span) -> Diagnostic {
    Diagnostic::error(format!("undefined name: {}", name))
        .with_span_label(
            file_path,
            span.start as usize,
            span.end as usize,
            format!("'{}' is not defined", name),
        )
        .with_help("Check the spelling, or add a definition for this name")
}

/// Create a diagnostic from a duplicate definition error
pub fn duplicate_definition_diagnostic(
    file_path: &str,
    name: &str,
    original: Span,
    duplicate: Span,
) -> Diagnostic {
    Diagnostic::error(format!("duplicate definition: {}", name))
        .with_span_label(
            file_path,
            duplicate.start as usize,
            duplicate.end as usize,
            format!("'{}' defined again here", name),
        )
        .with_label(
            file_path,
            original.start as usize,
            original.end as usize,
            "first definition here",
        )
        .with_help("Remove one of the definitions or rename one")
}

/// Create a diagnostic from an arity mismatch error
pub fn arity_mismatch_diagnostic(
    file_path: &str,
    expected: usize,
    got: usize,
    span: Span,
) -> Diagnostic {
    let message = if expected == 1 {
        format!("expected {} argument, found {}", expected, got)
    } else {
        format!("expected {} arguments, found {}", expected, got)
    };

    Diagnostic::error(format!("arity mismatch: {}", message))
        .with_span_label(file_path, span.start as usize, span.end as usize, message)
        .with_help("Check the operator definition for the correct number of arguments")
}

/// Create a diagnostic from a type error
pub fn type_error_diagnostic(file_path: &str, message: &str, span: Span) -> Diagnostic {
    Diagnostic::error(format!("type error: {}", message)).with_span_label(
        file_path,
        span.start as usize,
        span.end as usize,
        "type mismatch here",
    )
}

/// Create a diagnostic from a module not found error
pub fn module_not_found_diagnostic(file_path: &str, module_name: &str, span: Span) -> Diagnostic {
    Diagnostic::error(format!("module not found: {}", module_name))
        .with_span_label(
            file_path,
            span.start as usize,
            span.end as usize,
            format!("cannot find module '{}'", module_name),
        )
        .with_help("Check the module name and ensure the file exists")
}

/// Convert a core Error to a Diagnostic
impl crate::Error {
    /// Convert this error to a diagnostic
    pub fn to_diagnostic(&self, file_path: &str) -> Diagnostic {
        match self {
            crate::Error::Syntax { message, span } => {
                parse_error_diagnostic(file_path, message, span.start, span.end)
            }
            crate::Error::UndefinedName { name, span } => {
                undefined_name_diagnostic(file_path, name, *span)
            }
            crate::Error::DuplicateDefinition {
                name,
                original,
                duplicate,
            } => duplicate_definition_diagnostic(file_path, name, *original, *duplicate),
            crate::Error::Type { message, span } => {
                type_error_diagnostic(file_path, message, *span)
            }
            crate::Error::ArityMismatch {
                expected,
                got,
                span,
            } => arity_mismatch_diagnostic(file_path, *expected, *got, *span),
            crate::Error::ModuleNotFound { name, span } => {
                module_not_found_diagnostic(file_path, name, *span)
            }
            crate::Error::Io(err) => Diagnostic::error(format!("I/O error: {}", err)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;

    /// Helper to render diagnostic to string (strips ANSI color codes for snapshots)
    fn render_to_string(d: &Diagnostic, file_path: &str, source: &str) -> String {
        let mut buf = Vec::new();
        d.render(file_path, source, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        // Strip ANSI escape codes for deterministic snapshots
        strip_ansi_codes(&output)
    }

    /// Strip ANSI escape codes from a string
    fn strip_ansi_codes(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                // Skip the escape sequence: ESC [ ... m
                if chars.peek() == Some(&'[') {
                    chars.next(); // consume '['
                                  // Skip until we hit 'm'
                    while let Some(&nc) = chars.peek() {
                        chars.next();
                        if nc == 'm' {
                            break;
                        }
                    }
                    continue;
                }
            }
            result.push(c);
        }
        result
    }

    #[test]
    fn test_diagnostic_error() {
        let d = Diagnostic::error("test error");
        assert_eq!(d.severity, Severity::Error);
        assert_eq!(d.message, "test error");
    }

    #[test]
    fn test_diagnostic_with_span() {
        let d = Diagnostic::error("test").with_span("file.tla", 10, 20);
        assert!(d.span.is_some());
        let span = d.span.unwrap();
        assert_eq!(span.file, "file.tla");
        assert_eq!(span.start, 10);
        assert_eq!(span.end, 20);
    }

    #[test]
    fn test_diagnostic_render() {
        let source = "---- MODULE Test ----\nVARIABLE x\n====";

        let d = Diagnostic::error("undefined variable")
            .with_span_label("test.tla", 22, 31, "not defined")
            .with_help("Define the variable first");

        let mut buf = Vec::new();
        d.render("test.tla", source, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        // Verify the output contains expected content
        assert!(output.contains("undefined variable"));
        assert!(output.contains("not defined"));
    }

    #[test]
    fn test_parse_error_diagnostic() {
        let d = parse_error_diagnostic("test.tla", "unexpected token", 10, 15);
        assert_eq!(d.severity, Severity::Error);
        assert!(d.message.contains("syntax error"));
    }

    #[test]
    fn test_undefined_name_diagnostic() {
        let span = Span::new(crate::FileId(0), 10, 15);
        let d = undefined_name_diagnostic("test.tla", "foo", span);
        assert!(d.message.contains("undefined name"));
        assert!(d.help.is_some());
    }

    #[test]
    fn test_duplicate_definition_diagnostic() {
        let original = Span::new(crate::FileId(0), 10, 15);
        let duplicate = Span::new(crate::FileId(0), 30, 35);
        let d = duplicate_definition_diagnostic("test.tla", "foo", original, duplicate);
        assert!(d.message.contains("duplicate"));
        assert_eq!(d.labels.len(), 1); // Secondary label for original
    }

    // ============================================================================
    // SNAPSHOT TESTS - Error message format stability
    // These tests ensure error messages don't change unexpectedly.
    // ============================================================================

    #[test]
    fn snapshot_parse_error() {
        let source = "---- MODULE Test ----\nVARIABLE x == 1\n====";
        let d = parse_error_diagnostic("test.tla", "expected identifier, found '=='", 33, 35);
        assert_snapshot!(render_to_string(&d, "test.tla", source));
    }

    #[test]
    fn snapshot_undefined_name() {
        let source = "---- MODULE Test ----\nInit == foo = 0\n====";
        let span = Span::new(crate::FileId(0), 30, 33); // "foo"
        let d = undefined_name_diagnostic("test.tla", "foo", span);
        assert_snapshot!(render_to_string(&d, "test.tla", source));
    }

    #[test]
    fn snapshot_duplicate_definition() {
        let source = "---- MODULE Test ----\nfoo == 1\nfoo == 2\n====";
        let original = Span::new(crate::FileId(0), 22, 25); // first "foo"
        let duplicate = Span::new(crate::FileId(0), 31, 34); // second "foo"
        let d = duplicate_definition_diagnostic("test.tla", "foo", original, duplicate);
        assert_snapshot!(render_to_string(&d, "test.tla", source));
    }

    #[test]
    fn snapshot_arity_mismatch() {
        let source = "---- MODULE Test ----\nOp(x, y) == x + y\nInit == Op(1)\n====";
        let span = Span::new(crate::FileId(0), 48, 53); // "Op(1)"
        let d = arity_mismatch_diagnostic("test.tla", 2, 1, span);
        assert_snapshot!(render_to_string(&d, "test.tla", source));
    }

    #[test]
    fn snapshot_type_error() {
        let source = "---- MODULE Test ----\nInit == 1 + \"hello\"\n====";
        let span = Span::new(crate::FileId(0), 30, 42); // "1 + \"hello\""
        let d = type_error_diagnostic("test.tla", "cannot add integer and string", span);
        assert_snapshot!(render_to_string(&d, "test.tla", source));
    }

    #[test]
    fn snapshot_module_not_found() {
        let source = "---- MODULE Test ----\nEXTENDS Missing\n====";
        let span = Span::new(crate::FileId(0), 30, 37); // "Missing"
        let d = module_not_found_diagnostic("test.tla", "Missing", span);
        assert_snapshot!(render_to_string(&d, "test.tla", source));
    }

    #[test]
    fn snapshot_error_with_help_and_note() {
        let source = "---- MODULE Test ----\nInit == x' = 1\n====";
        let d = Diagnostic::error("primed variable 'x'' in Init")
            .with_span_label("test.tla", 30, 32, "primed variable not allowed here")
            .with_help("Init should not contain primed variables")
            .with_note(
                "Primed variables represent the next-state values and are used in Next, not Init",
            );
        assert_snapshot!(render_to_string(&d, "test.tla", source));
    }

    #[test]
    fn snapshot_multiline_context() {
        let source = r#"---- MODULE Test ----
VARIABLE x, y, z

Init ==
    /\ x = 0
    /\ y = undefined_var
    /\ z = 0
===="#;
        let span = Span::new(crate::FileId(0), 67, 80); // "undefined_var"
        let d = undefined_name_diagnostic("test.tla", "undefined_var", span);
        assert_snapshot!(render_to_string(&d, "test.tla", source));
    }
}
