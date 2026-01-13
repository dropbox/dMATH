//! Document state management
//!
//! Manages open documents and their state, including text content,
//! parsed AST, and elaboration results.

use ropey::Rope;
use tower_lsp::lsp_types::{Position, Range, Url};

/// A document being edited
#[derive(Debug, Clone)]
pub struct Document {
    /// Document URI
    pub uri: Url,
    /// Document version (increments on each edit)
    pub version: i32,
    /// Document text content (as rope for efficient edits)
    pub content: Rope,
    /// Language ID (should be "lean" or "lean4")
    pub language_id: String,
    /// Parsed AST (if available)
    pub parsed: Option<ParsedDocument>,
    /// Elaboration result (if available)
    pub elaborated: Option<ElaboratedDocument>,
    /// Incremental elaboration state (cached results)
    pub incremental_state: IncrementalState,
}

impl Document {
    /// Create a new document
    #[must_use]
    pub fn new(uri: Url, version: i32, content: String, language_id: String) -> Self {
        Self {
            uri,
            version,
            content: Rope::from_str(&content),
            language_id,
            parsed: None,
            elaborated: None,
            incremental_state: IncrementalState::default(),
        }
    }

    /// Get the full text content
    #[must_use]
    pub fn text(&self) -> String {
        self.content.to_string()
    }

    /// Get a line of text (0-indexed)
    #[must_use]
    pub fn line(&self, line_idx: usize) -> Option<String> {
        if line_idx < self.content.len_lines() {
            Some(self.content.line(line_idx).to_string())
        } else {
            None
        }
    }

    /// Get line count
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.content.len_lines()
    }

    /// Apply a text change
    pub fn apply_change(&mut self, range: Option<Range>, text: &str) {
        match range {
            Some(range) => {
                // Convert LSP range to byte offsets
                let start = self.position_to_offset(range.start);
                let end = self.position_to_offset(range.end);

                // Remove old text
                self.content.remove(start..end);
                // Insert new text
                self.content.insert(start, text);
            }
            None => {
                // Full document replacement - clear incremental state
                self.content = Rope::from_str(text);
                self.incremental_state = IncrementalState::default();
            }
        }

        // Invalidate parsed/elaborated results (but keep incremental state for partial edits)
        // The incremental state will be used to determine which commands need re-elaboration
        self.parsed = None;
        self.elaborated = None;
    }

    /// Convert LSP position to byte offset
    #[must_use]
    pub fn position_to_offset(&self, pos: Position) -> usize {
        let line_idx = pos.line as usize;
        if line_idx >= self.content.len_lines() {
            return self.content.len_bytes();
        }

        let line_start = self.content.line_to_byte(line_idx);
        let line = self.content.line(line_idx);

        // LSP positions are in UTF-16 code units, rope uses bytes
        let mut char_offset = 0;
        let mut utf16_offset = 0u32;

        for ch in line.chars() {
            if utf16_offset >= pos.character {
                break;
            }
            char_offset += ch.len_utf8();
            utf16_offset += ch.len_utf16() as u32;
        }

        line_start + char_offset
    }

    /// Convert byte offset to LSP position
    #[must_use]
    pub fn offset_to_position(&self, offset: usize) -> Position {
        let line = self
            .content
            .byte_to_line(offset.min(self.content.len_bytes()));
        let line_start = self.content.line_to_byte(line);
        let col_bytes = offset.saturating_sub(line_start);

        // Convert byte offset within line to UTF-16 code units
        let line_text = self.content.line(line);
        let mut utf16_col = 0u32;
        let mut byte_count = 0;

        for ch in line_text.chars() {
            if byte_count >= col_bytes {
                break;
            }
            byte_count += ch.len_utf8();
            utf16_col += ch.len_utf16() as u32;
        }

        Position {
            line: line as u32,
            character: utf16_col,
        }
    }
}

/// Parsed document (AST)
#[derive(Debug, Clone)]
pub struct ParsedDocument {
    /// Parse errors
    pub errors: Vec<ParseError>,
    /// Parsed commands/declarations
    pub commands: Vec<ParsedCommand>,
}

/// A parse error
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Byte offset start
    pub start: usize,
    /// Byte offset end
    pub end: usize,
    /// Error message
    pub message: String,
}

/// A parsed command/declaration
#[derive(Debug, Clone)]
pub struct ParsedCommand {
    /// Command kind
    pub kind: CommandKind,
    /// Byte offset start
    pub start: usize,
    /// Byte offset end
    pub end: usize,
    /// Name (if named declaration)
    pub name: Option<String>,
    /// Hash of the source text for this command (for incremental checking)
    pub content_hash: u64,
}

/// Kind of top-level command
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandKind {
    Definition,
    Theorem,
    Lemma,
    Example,
    Inductive,
    Structure,
    Class,
    Instance,
    Axiom,
    Variable,
    Universe,
    Import,
    Open,
    Namespace,
    Section,
    End,
    Other(String),
}

/// Elaborated document (type-checked)
#[derive(Debug, Clone)]
pub struct ElaboratedDocument {
    /// Type errors
    pub errors: Vec<TypeError>,
    /// Warnings (unused variables, deprecated features, etc.)
    pub warnings: Vec<Warning>,
    /// Declarations with types
    pub declarations: Vec<ElaboratedDecl>,
}

/// Cache entry for a single elaborated command
#[derive(Debug, Clone)]
pub struct ElaboratedCommandCache {
    /// Content hash of the source command (for cache invalidation)
    pub content_hash: u64,
    /// Type errors for this command
    pub errors: Vec<TypeError>,
    /// Warnings for this command
    pub warnings: Vec<Warning>,
    /// Elaborated declaration info (if applicable)
    pub declaration: Option<ElaboratedDecl>,
}

/// Incremental elaboration state
#[derive(Debug, Clone, Default)]
pub struct IncrementalState {
    /// Cached elaboration results keyed by command name or index
    /// Key is (name, content_hash) for named decls, or ("__anon_{idx}", content_hash) for anonymous
    pub cache: std::collections::HashMap<String, ElaboratedCommandCache>,
    /// Statistics for debugging
    pub stats: IncrementalStats,
}

/// Statistics about incremental checking
#[derive(Debug, Clone, Default)]
pub struct IncrementalStats {
    /// Number of commands in the document
    pub total_commands: usize,
    /// Number of commands that were re-elaborated
    pub elaborated_count: usize,
    /// Number of commands that used cached results
    pub cached_count: usize,
}

/// A warning (not a fatal error)
#[derive(Debug, Clone)]
pub struct Warning {
    /// Byte offset start
    pub start: usize,
    /// Byte offset end
    pub end: usize,
    /// Warning message
    pub message: String,
    /// Warning code (for categorization)
    pub code: WarningCode,
}

/// Categories of warnings
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarningCode {
    /// Variable declared but not used
    UnusedVariable,
    /// Import not needed
    UnusedImport,
    /// Using deprecated feature
    DeprecatedFeature,
    /// Code will never be reached
    UnreachableCode,
    /// Incomplete proof (sorry or admit)
    IncompleteProof,
    /// Other warning
    Other,
}

/// A type error
#[derive(Debug, Clone)]
pub struct TypeError {
    /// Byte offset start
    pub start: usize,
    /// Byte offset end
    pub end: usize,
    /// Error message
    pub message: String,
}

/// An elaborated declaration
#[derive(Debug, Clone)]
pub struct ElaboratedDecl {
    /// Declaration name
    pub name: String,
    /// Declaration type (pretty-printed)
    pub type_str: String,
    /// Byte offset start
    pub start: usize,
    /// Byte offset end
    pub end: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let uri = Url::parse("file:///test.lean").unwrap();
        let doc = Document::new(
            uri.clone(),
            1,
            "def x := 1\n".to_string(),
            "lean".to_string(),
        );

        assert_eq!(doc.version, 1);
        assert_eq!(doc.text(), "def x := 1\n");
        assert_eq!(doc.line_count(), 2); // includes trailing newline
    }

    #[test]
    fn test_document_line_access() {
        let uri = Url::parse("file:///test.lean").unwrap();
        let doc = Document::new(
            uri,
            1,
            "line1\nline2\nline3".to_string(),
            "lean".to_string(),
        );

        assert_eq!(doc.line(0), Some("line1\n".to_string()));
        assert_eq!(doc.line(1), Some("line2\n".to_string()));
        assert_eq!(doc.line(2), Some("line3".to_string()));
        assert_eq!(doc.line(3), None);
    }

    #[test]
    fn test_document_full_replace() {
        let uri = Url::parse("file:///test.lean").unwrap();
        let mut doc = Document::new(uri, 1, "old content".to_string(), "lean".to_string());

        doc.apply_change(None, "new content");

        assert_eq!(doc.text(), "new content");
    }

    #[test]
    fn test_document_partial_change() {
        let uri = Url::parse("file:///test.lean").unwrap();
        let mut doc = Document::new(uri, 1, "hello world".to_string(), "lean".to_string());

        // Replace "world" with "rust"
        let range = Range {
            start: Position {
                line: 0,
                character: 6,
            },
            end: Position {
                line: 0,
                character: 11,
            },
        };
        doc.apply_change(Some(range), "rust");

        assert_eq!(doc.text(), "hello rust");
    }

    #[test]
    fn test_position_offset_conversion() {
        let uri = Url::parse("file:///test.lean").unwrap();
        let doc = Document::new(
            uri,
            1,
            "line1\nline2\nline3".to_string(),
            "lean".to_string(),
        );

        // Start of file
        let pos = doc.offset_to_position(0);
        assert_eq!(pos.line, 0);
        assert_eq!(pos.character, 0);

        // Start of line 2
        let pos = doc.offset_to_position(6);
        assert_eq!(pos.line, 1);
        assert_eq!(pos.character, 0);

        // Middle of line 2
        let pos = doc.offset_to_position(9);
        assert_eq!(pos.line, 1);
        assert_eq!(pos.character, 3);
    }

    #[test]
    fn test_utf16_position_handling() {
        let uri = Url::parse("file:///test.lean").unwrap();
        // Emoji takes 4 bytes in UTF-8 but 2 code units in UTF-16
        let doc = Document::new(uri, 1, "a\u{1F600}b".to_string(), "lean".to_string());

        // 'a' is at position 0
        // emoji is at byte 1, but UTF-16 position 1-2
        // 'b' is at byte 5, but UTF-16 position 3
        let pos = doc.offset_to_position(5);
        assert_eq!(pos.character, 3); // 1 (a) + 2 (emoji surrogate pair)
    }

    #[test]
    fn test_command_kind_eq() {
        assert_eq!(CommandKind::Definition, CommandKind::Definition);
        assert_ne!(CommandKind::Definition, CommandKind::Theorem);
        assert_eq!(
            CommandKind::Other("foo".to_string()),
            CommandKind::Other("foo".to_string())
        );
    }
}
