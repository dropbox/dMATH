//! Document management for LSP
//!
//! Tracks open documents and their parsed/type-checked state.

use dashprove_usl::{parse, typecheck, ParseError, Spec, TypedSpec};
use std::collections::HashMap;
use std::sync::RwLock;
use tower_lsp::lsp_types::{Position, Range, Url};

/// A single USL document with its source and analysis results.
#[derive(Debug)]
pub struct Document {
    /// The document URI
    pub uri: Url,
    /// Document version (incremented on each change)
    pub version: i32,
    /// Raw source text
    pub text: String,
    /// Lines for position calculation
    lines: Vec<usize>,
    /// Parsed AST (if parsing succeeded)
    pub spec: Option<Spec>,
    /// Type-checked spec (if type checking succeeded)
    pub typed_spec: Option<TypedSpec>,
    /// Parse error (if parsing failed)
    pub parse_error: Option<ParseError>,
    /// Type errors (if type checking failed)
    pub type_errors: Vec<String>,
}

impl Document {
    /// Create a new document from source text.
    pub fn new(uri: Url, version: i32, text: String) -> Self {
        let mut doc = Self {
            uri,
            version,
            text: String::new(),
            lines: Vec::new(),
            spec: None,
            typed_spec: None,
            parse_error: None,
            type_errors: Vec::new(),
        };
        doc.update_text(text);
        doc
    }

    /// Update document text and re-analyze.
    pub fn update_text(&mut self, text: String) {
        self.text = text;
        self.compute_line_offsets();
        self.analyze();
    }

    /// Compute line start offsets for position conversion.
    fn compute_line_offsets(&mut self) {
        self.lines.clear();
        self.lines.push(0);
        for (i, ch) in self.text.char_indices() {
            if ch == '\n' {
                self.lines.push(i + 1);
            }
        }
    }

    /// Convert a (line, character) position to byte offset.
    #[must_use]
    pub fn position_to_offset(&self, line: u32, character: u32) -> usize {
        let line = line as usize;
        if line >= self.lines.len() {
            return self.text.len();
        }
        let line_start = self.lines[line];
        let line_text = if line + 1 < self.lines.len() {
            &self.text[line_start..self.lines[line + 1]]
        } else {
            &self.text[line_start..]
        };

        // character is in UTF-16 code units, convert to byte offset
        let mut byte_offset = 0;
        let mut utf16_offset = 0u32;
        for ch in line_text.chars() {
            if utf16_offset >= character {
                break;
            }
            utf16_offset += ch.len_utf16() as u32;
            byte_offset += ch.len_utf8();
        }

        line_start + byte_offset
    }

    /// Convert byte offset to (line, character) position.
    #[must_use]
    pub fn offset_to_position(&self, offset: usize) -> (u32, u32) {
        let offset = offset.min(self.text.len());

        // Find line
        let line = self
            .lines
            .iter()
            .position(|&start| start > offset)
            .map(|i| i.saturating_sub(1))
            .unwrap_or(self.lines.len().saturating_sub(1));

        // Calculate character (in UTF-16 code units)
        let line_start = self.lines[line];
        let text_before = &self.text[line_start..offset];
        let character: u32 = text_before.chars().map(|c| c.len_utf16() as u32).sum();

        (line as u32, character)
    }

    /// Get the word at a given position.
    #[must_use]
    pub fn word_at_position(&self, line: u32, character: u32) -> Option<&str> {
        let offset = self.position_to_offset(line, character);

        // Find word boundaries
        let start = self.text[..offset]
            .rfind(|c: char| !c.is_alphanumeric() && c != '_')
            .map(|i| i + 1)
            .unwrap_or(0);

        let end = self.text[offset..]
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .map(|i| offset + i)
            .unwrap_or(self.text.len());

        if start < end {
            Some(&self.text[start..end])
        } else {
            None
        }
    }

    /// Get the line text at a given line number.
    #[must_use]
    pub fn line_text(&self, line: u32) -> Option<&str> {
        let line = line as usize;
        if line >= self.lines.len() {
            return None;
        }
        let start = self.lines[line];
        let end = if line + 1 < self.lines.len() {
            self.lines[line + 1].saturating_sub(1)
        } else {
            self.text.len()
        };
        Some(&self.text[start..end])
    }

    /// Find the source range for an identifier, respecting word boundaries.
    #[must_use]
    pub fn find_identifier_range(&self, name: &str) -> Option<Range> {
        if name.is_empty() {
            return None;
        }

        let bytes = self.text.as_bytes();
        for (offset, _) in self.text.match_indices(name) {
            let before = offset
                .checked_sub(1)
                .and_then(|idx| bytes.get(idx).copied());
            let after = bytes.get(offset + name.len()).copied();

            let before_is_ident = before.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_');
            let after_is_ident = after.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_');

            if before_is_ident || after_is_ident {
                continue;
            }

            let (start_line, start_char) = self.offset_to_position(offset);
            let (end_line, end_char) = self.offset_to_position(offset + name.len());

            return Some(Range {
                start: Position::new(start_line, start_char),
                end: Position::new(end_line, end_char),
            });
        }

        None
    }

    /// Find all references to an identifier, respecting word boundaries.
    /// Returns ranges for all occurrences of the identifier in the document.
    #[must_use]
    pub fn find_all_references(&self, name: &str) -> Vec<Range> {
        if name.is_empty() {
            return Vec::new();
        }

        let bytes = self.text.as_bytes();
        let mut references = Vec::new();

        for (offset, _) in self.text.match_indices(name) {
            let before = offset
                .checked_sub(1)
                .and_then(|idx| bytes.get(idx).copied());
            let after = bytes.get(offset + name.len()).copied();

            let before_is_ident = before.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_');
            let after_is_ident = after.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_');

            if before_is_ident || after_is_ident {
                continue;
            }

            let (start_line, start_char) = self.offset_to_position(offset);
            let (end_line, end_char) = self.offset_to_position(offset + name.len());

            references.push(Range {
                start: Position::new(start_line, start_char),
                end: Position::new(end_line, end_char),
            });
        }

        references
    }

    /// Parse and type-check the document.
    fn analyze(&mut self) {
        self.spec = None;
        self.typed_spec = None;
        self.parse_error = None;
        self.type_errors.clear();

        // Parse
        match parse(&self.text) {
            Ok(spec) => {
                self.spec = Some(spec.clone());

                // Type check
                match typecheck(spec) {
                    Ok(typed) => {
                        self.typed_spec = Some(typed);
                    }
                    Err(e) => {
                        self.type_errors.push(e.to_string());
                    }
                }
            }
            Err(e) => {
                self.parse_error = Some(e);
            }
        }
    }
}

/// Thread-safe document store.
#[derive(Debug, Default)]
pub struct DocumentStore {
    documents: RwLock<HashMap<Url, Document>>,
}

impl DocumentStore {
    /// Create a new document store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a document.
    pub fn open(&self, uri: Url, version: i32, text: String) {
        let doc = Document::new(uri.clone(), version, text);
        let mut docs = self.documents.write().unwrap();
        docs.insert(uri, doc);
    }

    /// Update a document.
    pub fn update(&self, uri: &Url, version: i32, text: String) {
        let mut docs = self.documents.write().unwrap();
        if let Some(doc) = docs.get_mut(uri) {
            doc.version = version;
            doc.update_text(text);
        }
    }

    /// Close a document.
    pub fn close(&self, uri: &Url) {
        let mut docs = self.documents.write().unwrap();
        docs.remove(uri);
    }

    /// Get a document by URI (for reading).
    pub fn with_document<F, R>(&self, uri: &Url, f: F) -> Option<R>
    where
        F: FnOnce(&Document) -> R,
    {
        let docs = self.documents.read().unwrap();
        docs.get(uri).map(f)
    }

    /// Get all document URIs.
    pub fn all_uris(&self) -> Vec<Url> {
        let docs = self.documents.read().unwrap();
        docs.keys().cloned().collect()
    }

    /// Analyze all documents and collect results.
    ///
    /// Applies the given function to each open document and returns
    /// a vector of results, useful for workspace-wide operations.
    pub fn analyze_all<F, R>(&self, f: F) -> Vec<R>
    where
        F: Fn(&Document) -> R,
    {
        let docs = self.documents.read().unwrap();
        docs.values().map(f).collect()
    }

    /// Apply a function to every open document.
    pub fn for_each_document<F>(&self, mut f: F)
    where
        F: FnMut(&Document),
    {
        let docs = self.documents.read().unwrap();
        for doc in docs.values() {
            f(doc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_conversion() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo {\n    forall x: Bool . x\n}".to_string(),
        );

        // Line 0, char 0 = offset 0
        assert_eq!(doc.position_to_offset(0, 0), 0);
        // Line 0, char 7 = offset 7 ("theorem")
        assert_eq!(doc.position_to_offset(0, 7), 7);
        // Line 1, char 0 = offset 14 (after "theorem foo {\n")
        assert_eq!(doc.position_to_offset(1, 0), 14);

        // Reverse conversion
        assert_eq!(doc.offset_to_position(0), (0, 0));
        assert_eq!(doc.offset_to_position(7), (0, 7));
        assert_eq!(doc.offset_to_position(14), (1, 0));
    }

    #[test]
    fn test_word_at_position() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { forall x: Bool . x }".to_string(),
        );

        // "theorem" at position (0, 3)
        assert_eq!(doc.word_at_position(0, 3), Some("theorem"));
        // "foo" at position (0, 9)
        assert_eq!(doc.word_at_position(0, 9), Some("foo"));
        // "forall" at position (0, 16)
        assert_eq!(doc.word_at_position(0, 16), Some("forall"));
        // "Bool" at position (0, 25)
        assert_eq!(doc.word_at_position(0, 25), Some("Bool"));
    }

    #[test]
    fn test_document_analysis() {
        // Valid document
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem foo { forall x: Bool . x }".to_string(),
        );
        assert!(doc.spec.is_some());
        assert!(doc.typed_spec.is_some());
        assert!(doc.parse_error.is_none());
        assert!(doc.type_errors.is_empty());

        // Invalid syntax
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "theorem { }".to_string(), // missing name
        );
        assert!(doc.spec.is_none());
        assert!(doc.parse_error.is_some());
    }

    #[test]
    fn test_find_identifier_range() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Value = { id: Int }\ntheorem value_id { Value.id }\n".to_string(),
        );

        let range = doc
            .find_identifier_range("Value")
            .expect("Value should be found");
        assert_eq!(range.start.line, 0);
        assert_eq!(range.start.character, 5);
        assert_eq!(range.end.line, 0);
        assert_eq!(range.end.character, 10);

        // Should not match inside another identifier
        assert!(doc.find_identifier_range("Val").is_none());
    }

    #[test]
    fn test_document_store() {
        let store = DocumentStore::new();
        let uri = Url::parse("file:///test.usl").unwrap();

        store.open(uri.clone(), 1, "theorem foo { true }".to_string());

        let result = store.with_document(&uri, |doc| doc.version);
        assert_eq!(result, Some(1));

        store.update(&uri, 2, "theorem bar { false }".to_string());

        let result = store.with_document(&uri, |doc| doc.version);
        assert_eq!(result, Some(2));

        store.close(&uri);

        let result = store.with_document(&uri, |doc| doc.version);
        assert_eq!(result, None);
    }

    #[test]
    fn test_find_all_references() {
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type Value = { id: Int }\ntheorem value_check { forall v: Value . Value.id > 0 }"
                .to_string(),
        );

        // "Value" appears 3 times: definition, type annotation, and field access
        let refs = doc.find_all_references("Value");
        assert_eq!(refs.len(), 3);

        // First reference at line 0
        assert_eq!(refs[0].start.line, 0);
        assert_eq!(refs[0].start.character, 5);

        // Second reference at line 1 (type annotation)
        assert_eq!(refs[1].start.line, 1);

        // Third reference at line 1 (field access)
        assert_eq!(refs[2].start.line, 1);

        // Should not find partial matches
        let refs = doc.find_all_references("Val");
        assert_eq!(refs.len(), 0);

        // Should find "id" only at word boundaries (not inside "id:" pattern)
        let refs = doc.find_all_references("id");
        assert_eq!(refs.len(), 2); // In type def "id: Int" and in "Value.id"
    }

    // ========================================================================
    // Mutation-killing tests
    // ========================================================================

    #[test]
    fn test_position_to_offset_multiline_boundary() {
        // Tests line 74: `if line + 1 < self.lines.len()`
        // Mutant: replace < with >
        // This test ensures the boundary check correctly determines
        // whether we're on the last line or not
        //
        // Key insight: When mutant changes < to >, it makes the condition
        // false for middle lines when it should be true. This causes the
        // function to use text[line_start..] instead of text[line_start..lines[line+1]].
        // We need to test a character position beyond the end of a line but
        // within the next line's content - the mutant would incorrectly continue
        // counting into subsequent lines.

        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "ab\ncdefgh\nij".to_string(), // line0="ab", line1="cdefgh", line2="ij"
        );

        // Lines: 0 starts at 0, 1 starts at 3, 2 starts at 10
        // Line 0: "ab\n" (indices 0-2)
        // Line 1: "cdefgh\n" (indices 3-9)
        // Line 2: "ij" (indices 10-11)

        // Test character beyond end of line 0 (which is "ab" - only 2 chars)
        // Position (0, 5) should be clamped to end of line 0
        // Original: iterates through "ab\n", stops at offset 2 (newline or beyond)
        // Mutant: iterates through "ab\ncdefgh\nij", would continue to offset 5
        let offset = doc.position_to_offset(0, 5);
        // With correct behavior, we should stop at the end of line 0's content
        // Line 0 has "ab" (2 chars + newline), so requesting char 5 on line 0
        // should give us offset 3 (including newline) or 2 (excluding)
        // Actually, the loop iterates until utf16_offset >= character
        // For "ab\n": a(1), b(1), \n(1) = 3 utf16 units
        // So requesting char 5 would exhaust the line text and return line_start + byte_offset
        // = 0 + 3 = 3 (includes newline)
        //
        // But with mutant (using rest of file): a(1),b(1),\n(1),c(1),d(1) = 5 units at offset 5
        // So mutant would return 0 + 5 = 5
        assert_eq!(offset, 3); // Correct behavior: stops at end of line 0

        // Double-check middle line behavior
        let offset = doc.position_to_offset(1, 0);
        assert_eq!(offset, 3);

        // Test last line (line + 1 >= lines.len(), so we use text[line_start..])
        let offset = doc.position_to_offset(2, 0);
        assert_eq!(offset, 10);
    }

    #[test]
    fn test_find_identifier_range_underscore_boundary() {
        // Tests line 168: `before.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')`
        // Mutant: replace || with &&
        // The underscore is a valid identifier character, so identifiers with underscores
        // should not match partial segments

        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type my_value = { field: Int }\ntheorem my_value_check { true }".to_string(),
        );

        // "my_value" should be found at definition, not partial match
        let range = doc.find_identifier_range("my_value");
        assert!(range.is_some());
        let r = range.unwrap();
        assert_eq!(r.start.character, 5);
        assert_eq!(r.end.character, 13);

        // "value" should NOT be found because it's part of "my_value" and "my_value_check"
        // Both occurrences have underscore before or after
        let range = doc.find_identifier_range("value");
        assert!(range.is_none());

        // "my" should NOT be found - it's followed by underscore in "my_value"
        let range = doc.find_identifier_range("my");
        assert!(range.is_none());
    }

    #[test]
    fn test_find_all_references_underscore_boundary() {
        // Tests line 204: `before.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_')`
        // Mutant: replace || with &&
        // Same as above but for find_all_references

        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type foo_bar = { baz: Int }\ntheorem foo_bar_test { foo_bar.baz > 0 }".to_string(),
        );

        // "foo_bar" appears twice: definition and usage
        let refs = doc.find_all_references("foo_bar");
        assert_eq!(refs.len(), 2);

        // "bar" should NOT be found - preceded by underscore in both cases
        let refs = doc.find_all_references("bar");
        assert_eq!(refs.len(), 0);

        // "foo" should NOT be found - followed by underscore in both cases
        let refs = doc.find_all_references("foo");
        assert_eq!(refs.len(), 0);

        // Test with identifier that has underscore at end too
        let doc = Document::new(
            Url::parse("file:///test.usl").unwrap(),
            1,
            "type _private = { x: Int }".to_string(),
        );
        // "private" should NOT match because it's preceded by underscore
        let refs = doc.find_all_references("private");
        assert_eq!(refs.len(), 0);
        // "_private" should match
        let refs = doc.find_all_references("_private");
        assert_eq!(refs.len(), 1);
    }

    #[test]
    fn test_analyze_all_returns_results() {
        // Tests line 310: `docs.values().map(f).collect()`
        // Mutant: replace with vec![]
        // Ensures analyze_all actually collects results from documents

        let store = DocumentStore::new();
        let uri1 = Url::parse("file:///test1.usl").unwrap();
        let uri2 = Url::parse("file:///test2.usl").unwrap();
        let uri3 = Url::parse("file:///test3.usl").unwrap();

        store.open(uri1.clone(), 1, "theorem a { true }".to_string());
        store.open(uri2.clone(), 2, "theorem b { false }".to_string());
        store.open(uri3.clone(), 3, "theorem c { true }".to_string());

        // analyze_all should return results for all documents
        let versions: Vec<i32> = store.analyze_all(|doc| doc.version);
        assert_eq!(versions.len(), 3);

        // All versions should be present (order not guaranteed)
        let mut sorted = versions.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3]);

        // Test with empty store
        let empty_store = DocumentStore::new();
        let results: Vec<i32> = empty_store.analyze_all(|doc| doc.version);
        assert!(results.is_empty());
    }

    #[test]
    fn test_for_each_document_iterates() {
        // Tests line 319: `for doc in docs.values() { f(doc); }`
        // Mutant: replace with ()
        // Ensures for_each_document actually calls the function for each document

        use std::cell::Cell;

        let store = DocumentStore::new();
        let uri1 = Url::parse("file:///test1.usl").unwrap();
        let uri2 = Url::parse("file:///test2.usl").unwrap();

        store.open(uri1.clone(), 1, "theorem a { true }".to_string());
        store.open(uri2.clone(), 2, "theorem b { false }".to_string());

        // Count how many times the callback is invoked
        let count = Cell::new(0);
        store.for_each_document(|_doc| {
            count.set(count.get() + 1);
        });
        assert_eq!(count.get(), 2);

        // Collect all versions to verify actual iteration
        let mut versions = Vec::new();
        store.for_each_document(|doc| {
            versions.push(doc.version);
        });
        versions.sort();
        assert_eq!(versions, vec![1, 2]);

        // Test with empty store - callback should never be called
        let empty_store = DocumentStore::new();
        let mut called = false;
        empty_store.for_each_document(|_| {
            called = true;
        });
        assert!(!called);
    }
}
